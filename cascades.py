import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Callable, Tuple, Optional, Union
import warnings


def calculate_youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate the optimal threshold using Youden's index (J statistic).
    
    Youden's index = Sensitivity + Specificity - 1
    
    The threshold that maximizes this index is chosen as the optimal cutoff.
    This is equivalent to maximizing the vertical distance from the ROC curve
    to the chance diagonal.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_scores : array-like of shape (n_samples,)
        Predicted scores/probabilities.
        
    Returns
    -------
    optimal_threshold : float
        The threshold that maximizes Youden's index.
    """
    thresholds = np.unique(y_scores)
    
    thresholds = np.concatenate([
        [y_scores.min() - 1e-10],
        thresholds,
        [y_scores.max() + 1e-10]
    ])
    
    best_youden = -np.inf
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        
        youden = sensitivity + specificity - 1
        
        if youden > best_youden:
            best_youden = youden
            best_threshold = threshold
    
    return best_threshold


class NonCommitmentStrategy:
    """Base class for non-commitment region determination strategies."""
    
    def calculate_region(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        threshold: float) -> Tuple[float, float]:
        """
        Calculate non-commitment region boundaries.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels (0 or 1).
        y_pred : array-like of shape (n_samples,)
            Predicted scores/probabilities.
        threshold : float
            Classification threshold.
            
        Returns
        -------
        lower_bound : float
            Lower boundary of non-commitment region.
        upper_bound : float
            Upper boundary of non-commitment region.
        """
        raise NotImplementedError("Subclasses must implement calculate_region")
    
    def to_dict(self):
        """Serialize strategy to dictionary for JSON storage."""
        return {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__
        }
    
    @staticmethod
    def from_dict(data):
        """Reconstruct strategy from dictionary."""
        class_name = data.get('class')
        if class_name == 'FixedWidthStrategy':
            return FixedWidthStrategy(width=data.get('width', 0.1))
        elif class_name == 'AdaptiveF1Strategy':
            return AdaptiveF1Strategy(
                delta_step=data.get('delta_step', 0.01),
                penalization=data.get('penalization', 1.0)
            )
        elif class_name == 'AdaptiveNegativeF1Strategy':
            return AdaptiveNegativeF1Strategy(
                delta_step=data.get('delta_step', 0.01),
                penalization=data.get('penalization', 1.0)
            )
        elif class_name == 'AdaptiveCustomCostStrategy':
            # Cannot reconstruct custom lambda functions
            raise ValueError("AdaptiveCustomCostStrategy with custom functions cannot be serialized")
        else:
            raise ValueError(f"Unknown strategy class: {class_name}")


class FixedWidthStrategy(NonCommitmentStrategy):
    """Fixed width non-commitment region around threshold."""
    
    def __init__(self, width: float = 0.1):
        """
        Parameters
        ----------
        width : float, default=0.1
            Half-width of the non-commitment region.
        """
        self.width = width
    
    def calculate_region(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        threshold: float) -> Tuple[float, float]:
        return (threshold - self.width, threshold + self.width)
    
    def to_dict(self):
        """Serialize strategy to dictionary."""
        d = super().to_dict()
        d['width'] = self.width
        return d


class AdaptiveCustomCostStrategy(NonCommitmentStrategy):
    """
    Adaptive non-commitment region based on custom cost function optimization.
    
    This strategy allows users to define their own cost function that takes
    F1 score and fraction of points in the non-commitment region as inputs.
    The strategy searches for the delta that minimizes the provided cost function.
    
    Parameters
    ----------
    cost_function : callable
        A function that takes two arguments:
        - f1_score (float): F1 score for classifications outside the region
        - fraction_in_region (float): Fraction of points in non-commitment region
        Returns a float representing the cost to minimize.
        
    delta_step : float, default=0.01
        Step size for delta search.
        
    Examples
    --------
    >>> # Minimize 1/F1 + 2 * fraction^2 (stronger penalization)
    >>> def custom_cost(f1, frac):
    ...     return 1.0 / (f1 + 1e-10) + 2.0 * (frac ** 2)
    >>> strategy = AdaptiveCustomCostStrategy(cost_function=custom_cost)
    
    >>> # Minimize 1/F1 + fraction (linear penalization)
    >>> def linear_cost(f1, frac):
    ...     return 1.0 / (f1 + 1e-10) + frac
    >>> strategy = AdaptiveCustomCostStrategy(cost_function=linear_cost)
    
    >>> # Maximize F1 while keeping region small (equivalent to 1-F1 + fraction)
    >>> def balanced_cost(f1, frac):
    ...     return (1.0 - f1) + 0.5 * frac
    >>> strategy = AdaptiveCustomCostStrategy(cost_function=balanced_cost)
    """
    
    def __init__(self, cost_function: Callable[[float, float], float], 
                 delta_step: float = 0.01):
        """
        Parameters
        ----------
        cost_function : callable
            Function(f1_score, fraction_in_region) -> cost
        delta_step : float, default=0.01
            Step size for delta search.
        """
        self.cost_function = cost_function
        self.delta_step = delta_step
    
    def calculate_region(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        threshold: float) -> Tuple[float, float]:
        """
        Calculate non-commitment region by optimizing the custom cost function.
        
        The method creates prefix sums of class counts sorted by prediction
        scores, then searches for the optimal delta that minimizes the
        user-provided cost function.
        """
        # Create sorted data with prefixes (cumulative counts)
        prefixes = self._create_prefixes(y_true, y_pred)
        
        if len(prefixes) == 0:
            return (threshold, threshold)
        
        # Initialize search
        min_pred = min(prefixes.keys())
        max_delta = threshold - min_pred
        
        best_cost = np.inf
        best_delta = max_delta
        
        delta = max_delta
        n_total = len(y_true)
        
        # Search for optimal delta
        while delta > 0:
            beginning_value = self._get_key_for_value(prefixes, threshold - delta)
            end_value = self._get_key_for_value(prefixes, threshold + delta)
            
            # Calculate F1 score outside the non-commitment region
            f1 = self._f1_score(prefixes, beginning_value, end_value)
            
            # Calculate fraction of points in non-commitment region
            n_between = self._get_points_between(prefixes, beginning_value, end_value)
            frac_between = n_between / n_total
            
            # Evaluate custom cost function
            cost = self.cost_function(f1, frac_between)
            
            if cost < best_cost:
                best_cost = cost
                best_delta = delta
            
            delta -= self.delta_step
        
        return (threshold - best_delta, threshold + best_delta)
    
    def _create_prefixes(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Create prefix sums of class counts sorted by predictions."""
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        y_true_sorted = y_true[sorted_indices]
        
        unique_preds = np.unique(y_pred_sorted)
        prefixes = {}
        
        count_0, count_1 = 0, 0
        for pred_val in unique_preds:
            mask = y_pred_sorted == pred_val
            count_0 += np.sum((y_true_sorted == 0) & mask)
            count_1 += np.sum((y_true_sorted == 1) & mask)
            prefixes[pred_val] = (count_0, count_1)
        
        return prefixes
    
    def _get_key_for_value(self, prefixes: dict, value: float) -> float:
        """Find the largest key in prefixes that is <= value."""
        valid_keys = [k for k in prefixes.keys() if k <= value]
        if not valid_keys:
            return min(prefixes.keys())
        return max(valid_keys)
    
    def _f1_score(self, prefixes: dict, beginning: float, end: float) -> float:
        """Calculate F1 score for classifications outside [beginning, end]."""
        max_key = max(prefixes.keys())
        total_0, total_1 = prefixes[max_key]
        
        # Get counts at boundaries
        beg_0, beg_1 = prefixes.get(beginning, (0, 0))
        end_0, end_1 = prefixes.get(end, (0, 0))
        
        # TP: positive class points above end threshold
        tp = total_1 - end_1
        # FP: negative class points above end threshold  
        fp = total_0 - end_0
        # FN: positive class points below beginning threshold
        fn = beg_1
        
        eps = np.finfo(float).eps
        return 2 * tp / (2 * tp + fp + fn + eps)
    
    def _get_points_between(self, prefixes: dict, beginning: float, 
                           end: float) -> int:
        """Count total points between beginning and end."""
        beg_key = self._get_key_for_value(prefixes, beginning)
        end_key = self._get_key_for_value(prefixes, end)
        
        beg_0, beg_1 = prefixes.get(beg_key, (0, 0))
        end_0, end_1 = prefixes.get(end_key, (0, 0))
        
        n_0 = end_0 - beg_0
        n_1 = end_1 - beg_1
        
        return n_0 + n_1


class AdaptiveF1Strategy(AdaptiveCustomCostStrategy):
    """
    Adaptive non-commitment region based on F1-score optimization.
    
    The cost function used is:
    score = 1/F1 + penalization * (fraction_in_region)^2
    
    Parameters
    ----------
    delta_step : float, default=0.01
        Step size for delta search.
    penalization : float, default=1.0
        Penalization factor for fraction in non-commitment region.
        Use 0.5 for hard penalization, 1.0 for soft penalization.
        
    Examples
    --------
    >>> # Standard F1 optimization with soft penalization
    >>> strategy = AdaptiveF1Strategy(penalization=1.0)
    
    >>> # Hard penalization for smaller regions
    >>> strategy = AdaptiveF1Strategy(penalization=0.5)
    """
    
    def __init__(self, delta_step: float = 0.01, penalization: float = 1.0):
        """
        Parameters
        ----------
        delta_step : float, default=0.01
            Step size for delta search.
        penalization : float, default=1.0
            Penalization factor for fraction in non-commitment region.
        """
        self.penalization = penalization
        self.delta_step = delta_step
        
        # Define the cost function based on penalization parameter
        def f1_cost_function(f1: float, frac: float) -> float:
            eps = np.finfo(float).eps
            return 1.0 / (f1 + eps) + self.penalization * (frac ** 2)
        
        # Initialize parent with the F1-based cost function
        super().__init__(cost_function=f1_cost_function, delta_step=delta_step)
    
    def to_dict(self):
        """Serialize strategy to dictionary."""
        return {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'delta_step': self.delta_step,
            'penalization': self.penalization
        }


class AdaptiveNegativeF1Strategy(AdaptiveCustomCostStrategy):
    """
    Adaptive non-commitment region based on negative F1-score optimization.
    
    The cost function used is:
    score = -F1 + penalization * (fraction_in_region)^2
    
    Parameters
    ----------
    delta_step : float, default=0.01
        Step size for delta search.
    penalization : float, default=1.0
        Penalization factor for fraction in non-commitment region.
        Higher values create smaller regions, lower values allow larger regions.
        
    Examples
    --------
    >>> # Standard negative F1 optimization with moderate penalization
    >>> strategy = AdaptiveNegativeF1Strategy(penalization=1.0)
    
    >>> # Stronger penalization for smaller regions
    >>> strategy = AdaptiveNegativeF1Strategy(penalization=2.0)
    
    >>> # Weaker penalization for larger regions
    >>> strategy = AdaptiveNegativeF1Strategy(penalization=0.5)
    """
    
    def __init__(self, delta_step: float = 0.01, penalization: float = 1.0):
        """
        Parameters
        ----------
        delta_step : float, default=0.01
            Step size for delta search.
        penalization : float, default=1.0
            Penalization factor for fraction in non-commitment region.
        """
        self.penalization = penalization
        self.delta_step = delta_step
        
        # Define the cost function using negative F1
        def negative_f1_cost_function(f1: float, frac: float) -> float:
            return -f1 + self.penalization * (frac ** 2)
        
        # Initialize parent with the negative F1-based cost function
        super().__init__(cost_function=negative_f1_cost_function, delta_step=delta_step)
    
    def to_dict(self):
        """Serialize strategy to dictionary."""
        return {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'delta_step': self.delta_step,
            'penalization': self.penalization
        }


class ThreeWayDecisionCascadeClassifier(ClassifierMixin, BaseEstimator):
    """
    Three-Way Decision Cascade Classifier with Non-Commitment Regions.
    
    A cascade of binary classifiers or regressors where each subsequent model handles
    cases in the non-commitment region of previous ones. The system can return three
    types of decisions: positive class, negative class, or non-commitment (uncertain).
    
    When using regressors as base models, their continuous output is normalized to [0,1]
    and used as decision scores for binary classification. Thresholds are applied to
    these normalized scores to make final predictions.
    
    Parameters
    ----------
    base_classifiers : list of estimators
        List of scikit-learn compatible classifiers or regressors to form the cascade.
        - Classifiers must implement `fit` and `predict_proba` or `decision_function`.
        - Regressors must implement `fit` and `predict`.
        
    non_commitment_strategy : NonCommitmentStrategy or list, default=None
        Strategy for determining non-commitment regions. Can be:
        - A single NonCommitmentStrategy instance (used for all classifiers)
        - A list of NonCommitmentStrategy instances (one per classifier)
        - None (defaults to AdaptiveF1Strategy)
        
    threshold : float, 'auto', or 'youden', default='auto'
        Classification threshold for the first classifier.
        - If 'auto', uses 0.5 for probability-based classifiers.
        - If 'youden', calculates optimal threshold using Youden's index.
        - If float, uses the specified value.
        
    min_samples_for_next : int, default=10
        Minimum absolute number of samples required to train the next classifier.
        This is checked in addition to min_samples_percentage.
        
    min_samples_percentage : float, default=0.15
        Minimum percentage (0-1) of original training samples required to
        train the next classifier. If the number of samples in the 
        non-commitment region falls below this percentage of the original
        training set size, cascade training stops and remaining classifiers
        are not trained. Set to 0.0 to disable percentage-based truncation.
        
    return_noncommitment : bool, default=False
        If True, `predict` can return a third class label for non-commitment.
        If False, non-committed samples are assigned to the closest class
        based on the last classifier's output.
        
    noncommitment_label : int, default=-1
        Label to use for non-commitment decisions when return_noncommitment=True.
        
    random_state : int, default=None
        Random seed for reproducibility. If specified, this seed (and incremental
        values for subsequent levels) will be set on base classifiers that have
        a random_state parameter. This ensures reproducible results across runs
        while maintaining diversity between cascade levels.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels (binary classification: 0 and 1).
        
    cascade_ : list of fitted estimators
        The fitted classifiers in the cascade.
        
    thresholds_ : list of float
        Classification thresholds for each classifier in the cascade.
        
    non_commitment_regions_ : list of tuple
        Non-commitment region boundaries (lower, upper) for each classifier.
        
    n_cascade_levels_ : int
        Number of classifiers actually used in the cascade.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> 
    >>> # Create cascade with adaptive non-commitment regions
    >>> cascade = ThreeWayDecisionCascadeClassifier(
    ...     base_classifiers=[
    ...         RandomForestClassifier(n_estimators=10, random_state=42),
    ...         LogisticRegression(random_state=42),
    ...         RandomForestClassifier(n_estimators=20, random_state=42)
    ...     ],
    ...     non_commitment_strategy=AdaptiveF1Strategy(penalization=1.0)
    ... )
    >>> 
    >>> X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    >>> cascade.fit(X, y)
    >>> predictions = cascade.predict(X)
    >>> 
    >>> # Get predictions with cascade depth information
    >>> predictions, depths = cascade.predict_with_depth(X)
    """
    
    def __init__(self, 
                 base_classifiers: List[BaseEstimator],
                 non_commitment_strategy: Optional[Union[NonCommitmentStrategy, 
                                                        List[NonCommitmentStrategy]]] = None,
                 threshold: Union[float, str] = 'auto',
                 min_samples_for_next: int = 10,
                 min_samples_percentage: float = 0.15,
                 return_noncommitment: bool = False,
                 noncommitment_label: int = -1,
                 random_state: Optional[int] = None):
        
        self.base_classifiers = base_classifiers
        self.non_commitment_strategy = non_commitment_strategy
        self.threshold = threshold
        self.min_samples_for_next = min_samples_for_next
        self.min_samples_percentage = min_samples_percentage
        self.return_noncommitment = return_noncommitment
        self.noncommitment_label = noncommitment_label
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the cascade classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
            
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Check for binary classification
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("ThreeWayDecisionCascadeClassifier only supports "
                           "binary classification")
        
        if not (set(self.classes_) == {0, 1}):
            raise ValueError("Class labels must be 0 and 1")
        
        # Initialize strategies
        if self.non_commitment_strategy is None:
            strategies = [AdaptiveF1Strategy() for _ in self.base_classifiers]
        elif isinstance(self.non_commitment_strategy, list):
            if len(self.non_commitment_strategy) != len(self.base_classifiers):
                raise ValueError("Number of strategies must match number of "
                               "base classifiers")
            strategies = self.non_commitment_strategy
        else:
            strategies = [self.non_commitment_strategy for _ in self.base_classifiers]
        
        # Initialize cascade
        self.cascade_ = []
        self.thresholds_ = []
        self.non_commitment_regions_ = []
        self.strategies_ = strategies
        
        # Store original training set size for percentage-based truncation
        n_original_samples = len(X)
        min_samples_threshold = max(
            self.min_samples_for_next,
            int(n_original_samples * self.min_samples_percentage)
        )
        
        # Training data for current level
        X_current = X.copy()
        y_current = y.copy()
        
        for i, (base_clf, strategy) in enumerate(zip(self.base_classifiers, strategies)):
            # Check if we have enough samples (both absolute and percentage-based)
            n_current_samples = len(X_current)
            percentage_remaining = 100.0 * n_current_samples / n_original_samples
            
            if n_current_samples < min_samples_threshold:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {n_current_samples} samples "
                    f"remaining ({percentage_remaining:.1f}% of original {n_original_samples}), "
                    f"below threshold of {min_samples_threshold} samples "
                    f"({100.0 * self.min_samples_percentage:.1f}%)"
                )
                break
            
            # Check if we have at least 2 classes
            unique_classes = np.unique(y_current)
            if len(unique_classes) < 2:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {len(unique_classes)} class "
                    f"remaining in non-commitment region. Need at least 2 classes."
                )
                break
            
            # Clone and fit classifier
            clf = clone(base_clf)
            # Set random state if specified (use different seed for each level)
            if self.random_state is not None and hasattr(clf, 'random_state'):
                # Use a different but reproducible seed for each level
                clf.random_state = self.random_state + i
            clf.fit(X_current, y_current)
            
            # Get predictions on training data
            y_scores = self._get_scores(clf, X_current)
            
            # Determine threshold
            if i == 0:
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y_current, y_scores)
                elif self.threshold == 'auto':
                    threshold = 0.5  # Default for probabilities
                else:
                    threshold = self.threshold  # User-specified value
            else:
                # For subsequent levels, always use Youden if specified initially
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y_current, y_scores)
                else:
                    threshold = 0.5  # Default for probabilities
            
            # Calculate non-commitment region
            lower, upper = strategy.calculate_region(y_current, y_scores, threshold)
            
            # Adjust threshold to center of non-commitment region
            new_threshold = (lower + upper) / 2.0
            
            # Store classifier and parameters
            self.cascade_.append(clf)
            self.thresholds_.append(new_threshold)
            self.non_commitment_regions_.append((lower, upper))
            
            # Extract samples in non-commitment region for next level
            in_region = (y_scores >= lower) & (y_scores <= upper)
            
            if not np.any(in_region):
                break
            
            X_current = X_current[in_region]
            y_current = y_current[in_region]
        
        self.n_cascade_levels_ = len(self.cascade_)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels. If return_noncommitment=True, may include
            noncommitment_label for uncertain samples.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, np.nan)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold, (lower, upper)) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_)
        ):
            # Only process uncommitted samples
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            scores = self._get_scores(clf, X_remaining)
            
            # Classify samples outside non-commitment region
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            predictions[decided_idx] = (scores[decided_mask] >= threshold).astype(int)
            committed[decided_idx] = True
        
        # Handle remaining non-committed samples
        if self.return_noncommitment:
            predictions[~committed] = self.noncommitment_label
        else:
            # Assign to closest class based on last classifier
            if np.any(~committed):
                X_remaining = X[~committed]
                scores = self._get_scores(self.cascade_[-1], X_remaining)
                predictions[~committed] = (scores >= self.thresholds_[-1]).astype(int)
        
        return predictions.astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. For non-committed samples (when
            return_noncommitment=False), returns the probability from
            the last classifier.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        probas = np.full((n_samples, 2), np.nan)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold, (lower, upper)) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_)
        ):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            scores = self._get_scores(clf, X_remaining)
            
            # For decided samples, convert scores to probabilities
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            # Simple probability estimate
            probas[decided_idx, 1] = scores[decided_mask]
            probas[decided_idx, 0] = 1 - scores[decided_mask]
            committed[decided_idx] = True
        
        # Handle non-committed samples
        if np.any(~committed):
            X_remaining = X[~committed]
            scores = self._get_scores(self.cascade_[-1], X_remaining)
            probas[~committed, 1] = scores
            probas[~committed, 0] = 1 - scores
        
        return probas
    
    def predict_with_depth(self, X):
        """
        Predict class labels and cascade depth at which decision was made.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        depths : ndarray of shape (n_samples,)
            Cascade level (0-indexed) at which each prediction was made.
            Non-committed samples have depth = n_cascade_levels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, np.nan)
        depths = np.full(n_samples, self.n_cascade_levels_, dtype=int)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold, (lower, upper)) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_)
        ):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            scores = self._get_scores(clf, X_remaining)
            
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            predictions[decided_idx] = (scores[decided_mask] >= threshold).astype(int)
            depths[decided_idx] = i
            committed[decided_idx] = True
        
        # Handle non-committed samples
        if self.return_noncommitment:
            predictions[~committed] = self.noncommitment_label
        else:
            if np.any(~committed):
                X_remaining = X[~committed]
                scores = self._get_scores(self.cascade_[-1], X_remaining)
                predictions[~committed] = (scores >= self.thresholds_[-1]).astype(int)
        
        return predictions.astype(int), depths
    
    def get_cascade_statistics(self, X, y):
        """
        Get detailed statistics for each level of the cascade.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
            
        Returns
        -------
        stats : list of dict
            Statistics for each cascade level including:
            - 'level': cascade level (0-indexed)
            - 'n_decided': number of samples decided at this level
            - 'n_remaining': number of samples passed to next level
            - 'accuracy': accuracy on decided samples
            - 'threshold': classification threshold
            - 'region': non-commitment region boundaries
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)
        
        stats = []
        committed = np.zeros(len(X), dtype=bool)
        
        for i, (clf, threshold, (lower, upper)) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_)
        ):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            y_remaining = y[mask]
            scores = self._get_scores(clf, X_remaining)
            
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            if len(decided_idx) > 0:
                predictions = (scores[decided_mask] >= threshold).astype(int)
                accuracy = np.mean(predictions == y_remaining[decided_mask])
            else:
                accuracy = np.nan
            
            stats.append({
                'level': i,
                'n_decided': len(decided_idx),
                'n_remaining': np.sum(mask) - len(decided_idx),
                'accuracy': accuracy,
                'threshold': threshold,
                'region': (lower, upper)
            })
            
            committed[decided_idx] = True
        
        return stats
    
    def _get_scores(self, clf, X):
        """
        Get decision scores from a classifier or regressor.
        
        For classifiers: tries predict_proba first, then decision_function.
        For regressors: uses predict() output as scores, normalized to [0,1].
        """
        if hasattr(clf, 'predict_proba'):
            probas = clf.predict_proba(X)
            if probas.shape[1] == 2:
                return probas[:, 1]
            else:
                return probas[:, -1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X)
            def normalize(results):
                res = (results - results.min()) / (results.max() - results.min() + 1e-10)
                return res
            return normalize(scores)
        else:
            # Assume it's a regressor - use predict() output as scores
            scores = clf.predict(X)
            scores = np.asarray(scores).ravel()
            # Normalize to [0, 1]
            score_min, score_max = scores.min(), scores.max()
            if score_max - score_min > 1e-10:
                scores = (scores - score_min) / (score_max - score_min)
            else:
                # All scores are the same, return 0.5
                scores = np.full_like(scores, 0.5)
            return scores


class WeightedThreeWayDecisionCascadeClassifier(ThreeWayDecisionCascadeClassifier):
    """
    Weighted Three-Way Decision Cascade Classifier.
    
    Unlike the standard cascade that trains each level on progressively smaller subsets,
    this variant trains ALL levels on the complete dataset, using sample weighting to
    emphasize samples in the non-commitment region from previous levels.
    
    This approach solves the key problem of training data bias in cascades - each level
    sees the full data distribution rather than just the "hard" cases, while still
    focusing more attention on uncertain samples.
    
    Parameters
    ----------
    base_classifiers : list of BaseEstimator
        List of classifiers to use at each cascade level.
        
    non_commitment_strategy : NonCommitmentStrategy or list, optional
        Strategy for determining non-commitment regions.
        
    threshold : float or str, default='auto'
        Classification threshold. Can be 'youden', 'auto', or a float value.
        
    min_samples_for_next : int, default=10
        Minimum number of samples needed to continue to next level.
        
    min_samples_percentage : float, default=0.15
        Minimum percentage of original samples needed to continue.
        
    return_noncommitment : bool, default=False
        Whether to return non-committed samples.
        
    noncommitment_label : int, default=-1
        Label for non-committed samples if return_noncommitment=True.
        
    weight_multiplier : float, default=3.0
        How much to upweight samples in non-commitment region.
        Higher values focus more on uncertain cases.
        
    random_state : int, optional
        Random state for reproducibility.
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> cascade = WeightedThreeWayDecisionCascadeClassifier(
    ...     base_classifiers=[
    ...         RandomForestClassifier(n_estimators=25, random_state=42),
    ...         RandomForestClassifier(n_estimators=50, random_state=42),
    ...         RandomForestClassifier(n_estimators=100, random_state=42)
    ...     ],
    ...     weight_multiplier=3.0,
    ...     non_commitment_strategy=AdaptiveF1Strategy(penalization=0.5)
    ... )
    >>> 
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>> cascade.fit(X, y)
    >>> predictions = cascade.predict(X)
    """
    
    def __init__(self, 
                 base_classifiers: List[BaseEstimator],
                 non_commitment_strategy: Optional[Union[NonCommitmentStrategy, 
                                                        List[NonCommitmentStrategy]]] = None,
                 threshold: Union[float, str] = 'auto',
                 min_samples_for_next: int = 10,
                 min_samples_percentage: float = 0.15,
                 return_noncommitment: bool = False,
                 noncommitment_label: int = -1,
                 weight_multiplier: float = 3.0,
                 random_state: Optional[int] = None):
        
        super().__init__(
            base_classifiers=base_classifiers,
            non_commitment_strategy=non_commitment_strategy,
            threshold=threshold,
            min_samples_for_next=min_samples_for_next,
            min_samples_percentage=min_samples_percentage,
            return_noncommitment=return_noncommitment,
            noncommitment_label=noncommitment_label,
            random_state=random_state
        )
        self.weight_multiplier = weight_multiplier
    
    def fit(self, X, y):
        """
        Fit the weighted cascade classifier.
        
        All levels are trained on the complete dataset, but samples in the
        non-commitment region from previous levels receive higher weight.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
            
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Check for binary classification
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("WeightedThreeWayDecisionCascadeClassifier only supports "
                           "binary classification")
        
        if not (set(self.classes_) == {0, 1}):
            raise ValueError("Class labels must be 0 and 1")
        
        # Initialize strategies
        if self.non_commitment_strategy is None:
            strategies = [AdaptiveF1Strategy() for _ in self.base_classifiers]
        elif isinstance(self.non_commitment_strategy, list):
            if len(self.non_commitment_strategy) != len(self.base_classifiers):
                raise ValueError("Number of strategies must match number of "
                               "base classifiers")
            strategies = self.non_commitment_strategy
        else:
            strategies = [self.non_commitment_strategy for _ in self.base_classifiers]
        
        # Initialize cascade
        self.cascade_ = []
        self.thresholds_ = []
        self.non_commitment_regions_ = []
        self.strategies_ = strategies
        
        # Store original training set size for percentage-based truncation
        n_original_samples = len(X)
        min_samples_threshold = max(
            self.min_samples_for_next,
            int(n_original_samples * self.min_samples_percentage)
        )
        
        # Initialize sample weights (all equal at first)
        sample_weights = np.ones(len(X))
        
        # Track which samples are still uncommitted
        uncommitted_mask = np.ones(len(X), dtype=bool)
        
        for i, (base_clf, strategy) in enumerate(zip(self.base_classifiers, strategies)):
            # Check if we have enough uncommitted samples
            n_uncommitted = np.sum(uncommitted_mask)
            percentage_uncommitted = 100.0 * n_uncommitted / n_original_samples
            
            if n_uncommitted < min_samples_threshold:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {n_uncommitted} uncommitted samples "
                    f"({percentage_uncommitted:.1f}% of original {n_original_samples}), "
                    f"below threshold of {min_samples_threshold} samples"
                )
                break
            
            # Check if we have at least 2 classes in uncommitted samples
            unique_classes = np.unique(y[uncommitted_mask])
            if len(unique_classes) < 2:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {len(unique_classes)} class "
                    f"in uncommitted samples. Need at least 2 classes."
                )
                break
            
            # Clone classifier
            clf = clone(base_clf)
            
            # Set random state if specified (use different seed for each level)
            if self.random_state is not None and hasattr(clf, 'random_state'):
                clf.random_state = self.random_state + i
            
            # Use bootstrap sampling with replacement based on weights
            # Normalize weights to probabilities
            sampling_probs = sample_weights / sample_weights.sum()
            
            # Sample with replacement
            indices = np.random.RandomState(self.random_state + i if self.random_state else None)\
                .choice(len(X), size=len(X), replace=True, p=sampling_probs)
            
            clf.fit(X[indices], y[indices])
            
            # Get predictions on ALL training data
            y_scores = self._get_scores(clf, X)
            
            # Calculate non-commitment region based on ALL data
            # (classifier was trained on all data with weights, so NC region should reflect that)
            
            # Determine threshold
            if i == 0:
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y, y_scores)
                elif self.threshold == 'auto':
                    threshold = 0.5
                else:
                    threshold = self.threshold
            else:
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y, y_scores)
                else:
                    threshold = 0.5
            
            # Calculate non-commitment region from ALL samples
            lower, upper = strategy.calculate_region(y, y_scores, threshold)
            
            # Adjust threshold to center of non-commitment region
            new_threshold = (lower + upper) / 2.0
            
            # Store classifier and parameters
            self.cascade_.append(clf)
            self.thresholds_.append(new_threshold)
            self.non_commitment_regions_.append((lower, upper))
            
            # Update uncommitted mask and weights for next level
            # Samples in non-commitment region remain uncommitted
            in_nc_region = (y_scores >= lower) & (y_scores <= upper)
            uncommitted_mask = uncommitted_mask & in_nc_region
            
            # If no samples remain uncommitted, stop
            if not np.any(uncommitted_mask):
                break
            
            # Update sample weights: upweight uncommitted samples for next level
            sample_weights = np.where(uncommitted_mask, 
                                     self.weight_multiplier, 
                                     1.0)
            
            # Normalize weights to sum to n_samples (keeps same effective sample size)
            sample_weights = sample_weights * len(X) / sample_weights.sum()
        
        self.n_cascade_levels_ = len(self.cascade_)
        
        return self


class FullDataCascadeClassifier(ThreeWayDecisionCascadeClassifier):
    """
    Full Data Cascade Classifier - trains all levels on complete dataset.
    
    This cascade variant trains ALL classifiers on the complete dataset without
    any subsetting or sample weighting. Each level uses a different model complexity
    (e.g., RF with 25, 50, 75, 100 estimators), but all see the same training data.
    
    At prediction time, samples still cascade through levels (early exit if committed),
    but the key difference is that training is unbiased - every classifier sees the
    full data distribution.
    
    This tests whether the training data bias in classical cascades is the primary
    cause of accuracy degradation, or if the cascade architecture itself has
    fundamental limitations.
    
    Parameters
    ----------
    base_classifiers : list of BaseEstimator
        List of classifiers to use at each cascade level.
        
    non_commitment_strategy : NonCommitmentStrategy or list, optional
        Strategy for determining non-commitment regions.
        
    threshold : float or str, default='auto'
        Classification threshold. Can be 'youden', 'auto', or a float value.
        
    min_samples_for_next : int, default=10
        Minimum number of samples needed to continue to next level.
        
    min_samples_percentage : float, default=0.15
        Minimum percentage of original samples needed to continue.
        
    return_noncommitment : bool, default=False
        Whether to return non-committed samples.
        
    noncommitment_label : int, default=-1
        Label for non-committed samples if return_noncommitment=True.
        
    random_state : int, optional
        Random state for reproducibility.
    """
    
    def __init__(self,
                 base_classifiers,
                 non_commitment_strategy=None,
                 threshold='auto',
                 min_samples_for_next=10,
                 min_samples_percentage=0.15,
                 return_noncommitment=False,
                 noncommitment_label=-1,
                 random_state=None):
        super().__init__(
            base_classifiers=base_classifiers,
            non_commitment_strategy=non_commitment_strategy,
            threshold=threshold,
            min_samples_for_next=min_samples_for_next,
            min_samples_percentage=min_samples_percentage,
            return_noncommitment=return_noncommitment,
            noncommitment_label=noncommitment_label,
            random_state=random_state
        )
    
    def fit(self, X, y):
        """
        Fit the cascade by training all classifiers on the FULL dataset.
        
        Unlike classical cascade (trains on subsets) and weighted cascade 
        (uses sample weights), this simply trains each classifier on all data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("FullDataCascadeClassifier only supports binary classification")
        
        if not (set(self.classes_) == {0, 1}):
            raise ValueError("Class labels must be 0 and 1")
        
        # Initialize strategies
        if self.non_commitment_strategy is None:
            strategies = [AdaptiveF1Strategy() for _ in self.base_classifiers]
        elif isinstance(self.non_commitment_strategy, list):
            if len(self.non_commitment_strategy) != len(self.base_classifiers):
                raise ValueError("Number of strategies must match number of base classifiers")
            strategies = self.non_commitment_strategy
        else:
            strategies = [self.non_commitment_strategy for _ in self.base_classifiers]
        
        # Initialize cascade
        self.cascade_ = []
        self.thresholds_ = []
        self.non_commitment_regions_ = []
        self.strategies_ = strategies
        
        # Store original training set size for percentage-based truncation
        n_original_samples = len(X)
        min_samples_threshold = max(
            self.min_samples_for_next,
            int(n_original_samples * self.min_samples_percentage)
        )
        
        # Track which samples are still uncommitted (used to determine NC regions)
        uncommitted_mask = np.ones(len(X), dtype=bool)
        
        for i, (base_clf, strategy) in enumerate(zip(self.base_classifiers, strategies)):
            # Check if we have enough uncommitted samples
            n_uncommitted = np.sum(uncommitted_mask)
            percentage_uncommitted = 100.0 * n_uncommitted / n_original_samples
            
            if n_uncommitted < min_samples_threshold:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {n_uncommitted} uncommitted samples "
                    f"({percentage_uncommitted:.1f}% of original {n_original_samples}), "
                    f"below threshold of {min_samples_threshold} samples"
                )
                break
            
            # Check if we have at least 2 classes in uncommitted samples
            if len(np.unique(y[uncommitted_mask])) < 2:
                warnings.warn(f"Stopping cascade at level {i}: only 1 class in uncommitted samples")
                break
            
            # KEY DIFFERENCE: Train on ALL data, not just uncommitted
            clf = clone(base_clf)
            clf.fit(X, y)  # Train on complete dataset
            
            # Get probability scores on ALL samples
            if hasattr(clf, "predict_proba"):
                y_scores = clf.predict_proba(X)[:, 1]
            else:
                y_scores = clf.decision_function(X)
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
            
            # Calculate non-commitment region based on ALL data
            # (not just uncommitted samples - this is the key difference from classical cascade)
            # The current classifier sees all data, so NC region should be based on all data
            
            # Calculate threshold
            if hasattr(clf, "predict_proba"):
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y, y_scores)
                elif self.threshold == 'auto':
                    threshold = 0.5
                else:
                    threshold = self.threshold
            else:
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y, y_scores)
                else:
                    threshold = 0.5
            
            # Calculate non-commitment region from ALL samples
            lower, upper = strategy.calculate_region(y, y_scores, threshold)
            
            # Adjust threshold to center of non-commitment region
            new_threshold = (lower + upper) / 2.0
            
            # Store classifier and parameters
            self.cascade_.append(clf)
            self.thresholds_.append(new_threshold)
            self.non_commitment_regions_.append((lower, upper))
            
            # Update uncommitted mask for next level
            # Samples in NC region AND still uncommitted from previous levels remain uncommitted
            in_nc_region = (y_scores >= lower) & (y_scores <= upper)
            uncommitted_mask = uncommitted_mask & in_nc_region
            
            # If no samples remain uncommitted, stop
            if not np.any(uncommitted_mask):
                break
        
        self.n_cascade_levels_ = len(self.cascade_)
        
        return self


class ProgressiveFeatureCascadeClassifier(ThreeWayDecisionCascadeClassifier):
    """
    Progressive Feature Cascade Classifier with incremental feature inclusion.
    
    This variant of the three-way cascade progressively adds more features at each level.
    The first classifier uses only the most informative features (based on variance),
    and each subsequent level adds more features, allowing for increasingly complex
    decision boundaries while maintaining computational efficiency.
    
    Parameters
    ----------
    base_classifiers : list of estimators
        List of scikit-learn compatible classifiers to form the cascade.
        
    feature_percentages : list of float, optional
        Percentage of features (0-1) to use at each cascade level.
        If None, defaults to [0.2, 0.4, 0.6, 0.8, 1.0] for up to 5 levels.
        Length should match or exceed the number of base_classifiers.
        
    feature_selection_method : {'variance', 'mutual_info', 'random'}, default='variance'
        Method to rank features:
        - 'variance': Select features with highest variance
        - 'mutual_info': Select features with highest mutual information with target
        - 'random': Select features randomly (for baseline comparison)
        
    non_commitment_strategy : NonCommitmentStrategy or list, default=None
        Strategy for determining non-commitment regions.
        
    threshold : float, 'auto', or 'youden', default='auto'
        Classification threshold for the first classifier.
        
    min_samples_for_next : int, default=10
        Minimum absolute number of samples required to train the next classifier.
        
    min_samples_percentage : float, default=0.15
        Minimum percentage (0-1) of original training samples required.
        
    return_noncommitment : bool, default=False
        If True, can return non-commitment label.
        
    noncommitment_label : int, default=-1
        Label to use for non-commitment decisions.
        
    random_state : int, default=None
        Random seed for reproducibility.
        
    Attributes
    ----------
    feature_indices_ : list of ndarray
        Indices of features used at each cascade level.
        
    feature_percentages_used_ : list of float
        Actual percentage of features used at each level.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> cascade = ProgressiveFeatureCascadeClassifier(
    ...     base_classifiers=[
    ...         LogisticRegression(),
    ...         RandomForestClassifier(n_estimators=10),
    ...         RandomForestClassifier(n_estimators=50)
    ...     ],
    ...     feature_percentages=[0.2, 0.5, 1.0],
    ...     non_commitment_strategy=AdaptiveF1Strategy()
    ... )
    >>> cascade.fit(X_train, y_train)
    >>> predictions = cascade.predict(X_test)
    """
    
    def __init__(self,
                 base_classifiers: List[BaseEstimator],
                 feature_percentages: Optional[List[float]] = None,
                 feature_selection_method: str = 'variance',
                 non_commitment_strategy: Optional[Union[NonCommitmentStrategy, 
                                                        List[NonCommitmentStrategy]]] = None,
                 threshold: Union[float, str] = 'auto',
                 min_samples_for_next: int = 10,
                 min_samples_percentage: float = 0.15,
                 return_noncommitment: bool = False,
                 noncommitment_label: int = -1,
                 random_state: Optional[int] = None):
        
        super().__init__(
            base_classifiers=base_classifiers,
            non_commitment_strategy=non_commitment_strategy,
            threshold=threshold,
            min_samples_for_next=min_samples_for_next,
            min_samples_percentage=min_samples_percentage,
            return_noncommitment=return_noncommitment,
            noncommitment_label=noncommitment_label,
            random_state=random_state
        )
        
        self.feature_percentages = feature_percentages
        self.feature_selection_method = feature_selection_method
    
    def fit(self, X, y):
        """
        Fit the progressive feature cascade classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
            
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Check for binary classification
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("ProgressiveFeatureCascadeClassifier only supports "
                           "binary classification")
        
        if not (set(self.classes_) == {0, 1}):
            raise ValueError("Class labels must be 0 and 1")
        
        n_features = X.shape[1]
        
        # Set default feature percentages if not provided
        if self.feature_percentages is None:
            default_percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
            self.feature_percentages = default_percentages[:len(self.base_classifiers)]
        
        if len(self.feature_percentages) < len(self.base_classifiers):
            raise ValueError("feature_percentages must have at least as many "
                           "elements as base_classifiers")
        
        # Rank features by importance
        feature_ranking = self._rank_features(X, y)
        
        # Determine feature subsets for each level
        self.feature_indices_ = []
        self.feature_percentages_used_ = []
        
        for pct in self.feature_percentages[:len(self.base_classifiers)]:
            n_features_to_use = max(1, int(n_features * pct))
            # Ensure we don't exceed total features
            n_features_to_use = min(n_features_to_use, n_features)
            
            # Select top features
            selected_features = feature_ranking[:n_features_to_use]
            self.feature_indices_.append(selected_features)
            self.feature_percentages_used_.append(n_features_to_use / n_features)
        
        # Initialize strategies
        if self.non_commitment_strategy is None:
            strategies = [AdaptiveF1Strategy() for _ in self.base_classifiers]
        elif isinstance(self.non_commitment_strategy, list):
            if len(self.non_commitment_strategy) != len(self.base_classifiers):
                raise ValueError("Number of strategies must match number of "
                               "base classifiers")
            strategies = self.non_commitment_strategy
        else:
            strategies = [self.non_commitment_strategy for _ in self.base_classifiers]
        
        # Initialize cascade
        self.cascade_ = []
        self.thresholds_ = []
        self.non_commitment_regions_ = []
        self.strategies_ = strategies
        
        # Store original training set size
        n_original_samples = len(X)
        min_samples_threshold = max(
            self.min_samples_for_next,
            int(n_original_samples * self.min_samples_percentage)
        )
        
        # Training data for current level
        X_current = X.copy()
        y_current = y.copy()
        
        for i, (base_clf, strategy, feature_idx) in enumerate(
            zip(self.base_classifiers, strategies, self.feature_indices_)
        ):
            # Check sample constraints
            n_current_samples = len(X_current)
            percentage_remaining = 100.0 * n_current_samples / n_original_samples
            
            if n_current_samples < min_samples_threshold:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {n_current_samples} samples "
                    f"remaining ({percentage_remaining:.1f}% of original {n_original_samples}), "
                    f"below threshold of {min_samples_threshold} samples"
                )
                break
            
            # Check class diversity
            unique_classes = np.unique(y_current)
            if len(unique_classes) < 2:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {len(unique_classes)} class "
                    f"remaining in non-commitment region"
                )
                break
            
            # Select features for this level
            X_level = X_current[:, feature_idx]
            
            # Clone and fit classifier
            clf = clone(base_clf)
            if self.random_state is not None and hasattr(clf, 'random_state'):
                clf.random_state = self.random_state + i
            clf.fit(X_level, y_current)
            
            # Get predictions on training data
            y_scores = self._get_scores(clf, X_level)
            
            # Determine threshold
            if i == 0:
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y_current, y_scores)
                elif self.threshold == 'auto':
                    threshold = 0.5
                else:
                    threshold = self.threshold
            else:
                if self.threshold == 'youden':
                    threshold = calculate_youden_threshold(y_current, y_scores)
                else:
                    threshold = 0.5
            
            # Calculate non-commitment region
            lower, upper = strategy.calculate_region(y_current, y_scores, threshold)
            new_threshold = (lower + upper) / 2.0
            
            # Store classifier and parameters
            self.cascade_.append(clf)
            self.thresholds_.append(new_threshold)
            self.non_commitment_regions_.append((lower, upper))
            
            # Extract samples in non-commitment region for next level
            in_region = (y_scores >= lower) & (y_scores <= upper)
            
            if not np.any(in_region):
                break
            
            X_current = X_current[in_region]
            y_current = y_current[in_region]
        
        self.n_cascade_levels_ = len(self.cascade_)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using progressive feature subsets.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, np.nan)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold, (lower, upper), feature_idx) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_, 
                self.feature_indices_)
        ):
            # Only process uncommitted samples
            mask = ~committed
            if not np.any(mask):
                break
            
            # Select features for this level
            X_remaining = X[mask][:, feature_idx]
            scores = self._get_scores(clf, X_remaining)
            
            # Classify samples outside non-commitment region
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            predictions[decided_idx] = (scores[decided_mask] >= threshold).astype(int)
            committed[decided_idx] = True
        
        # Handle remaining non-committed samples
        if self.return_noncommitment:
            predictions[~committed] = self.noncommitment_label
        else:
            if np.any(~committed) and len(self.cascade_) > 0:
                # Use feature indices from the last trained level
                last_feature_idx = self.feature_indices_[len(self.cascade_) - 1]
                X_remaining = X[~committed][:, last_feature_idx]
                scores = self._get_scores(self.cascade_[-1], X_remaining)
                predictions[~committed] = (scores >= self.thresholds_[-1]).astype(int)
        
        return predictions.astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using progressive feature subsets.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        probas = np.full((n_samples, 2), np.nan)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold, (lower, upper), feature_idx) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_,
                self.feature_indices_)
        ):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask][:, feature_idx]
            scores = self._get_scores(clf, X_remaining)
            
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            probas[decided_idx, 1] = scores[decided_mask]
            probas[decided_idx, 0] = 1 - scores[decided_mask]
            committed[decided_idx] = True
        
        # Handle non-committed samples
        if np.any(~committed) and len(self.cascade_) > 0:
            # Use feature indices from the last trained level
            last_feature_idx = self.feature_indices_[len(self.cascade_) - 1]
            X_remaining = X[~committed][:, last_feature_idx]
            scores = self._get_scores(self.cascade_[-1], X_remaining)
            probas[~committed, 1] = scores
            probas[~committed, 0] = 1 - scores
        
        return probas
    
    def predict_with_depth(self, X):
        """
        Predict class labels and cascade depth with progressive features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        depths : ndarray of shape (n_samples,)
            Cascade level at which each prediction was made.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, np.nan)
        depths = np.full(n_samples, self.n_cascade_levels_, dtype=int)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold, (lower, upper), feature_idx) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_,
                self.feature_indices_)
        ):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask][:, feature_idx]
            scores = self._get_scores(clf, X_remaining)
            
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            predictions[decided_idx] = (scores[decided_mask] >= threshold).astype(int)
            depths[decided_idx] = i
            committed[decided_idx] = True
        
        # Handle non-committed samples
        if self.return_noncommitment:
            predictions[~committed] = self.noncommitment_label
        else:
            if np.any(~committed) and len(self.cascade_) > 0:
                # Use feature indices from the last trained level
                last_feature_idx = self.feature_indices_[len(self.cascade_) - 1]
                X_remaining = X[~committed][:, last_feature_idx]
                scores = self._get_scores(self.cascade_[-1], X_remaining)
                predictions[~committed] = (scores >= self.thresholds_[-1]).astype(int)
        
        return predictions.astype(int), depths
    
    def get_cascade_statistics(self, X, y):
        """
        Get detailed statistics for each level including feature information.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
            
        Returns
        -------
        stats : list of dict
            Statistics for each cascade level including feature usage.
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)
        
        stats = []
        committed = np.zeros(len(X), dtype=bool)
        
        for i, (clf, threshold, (lower, upper), feature_idx) in enumerate(
            zip(self.cascade_, self.thresholds_, self.non_commitment_regions_,
                self.feature_indices_)
        ):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask][:, feature_idx]
            y_remaining = y[mask]
            scores = self._get_scores(clf, X_remaining)
            
            decided_mask = (scores < lower) | (scores > upper)
            decided_idx = np.where(mask)[0][decided_mask]
            
            if len(decided_idx) > 0:
                predictions = (scores[decided_mask] >= threshold).astype(int)
                accuracy = np.mean(predictions == y_remaining[decided_mask])
            else:
                accuracy = np.nan
            
            stats.append({
                'level': i,
                'n_decided': len(decided_idx),
                'n_remaining': np.sum(mask) - len(decided_idx),
                'accuracy': accuracy,
                'threshold': threshold,
                'region': (lower, upper),
                'n_features': len(feature_idx),
                'feature_percentage': self.feature_percentages_used_[i],
                'feature_indices': feature_idx
            })
            
            committed[decided_idx] = True
        
        return stats
    
    def _rank_features(self, X, y):
        """
        Rank features by importance.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        ranked_indices : ndarray
            Indices of features sorted by importance (most important first).
        """
        if self.feature_selection_method == 'variance':
            # Calculate variance for each feature
            feature_variance = np.var(X, axis=0)
            ranked_indices = np.argsort(feature_variance)[::-1]
            
        elif self.feature_selection_method == 'mutual_info':
            # Use mutual information with target
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            ranked_indices = np.argsort(mi_scores)[::-1]
            
        elif self.feature_selection_method == 'random':
            # Random feature selection
            n_features = X.shape[1]
            rng = np.random.RandomState(self.random_state)
            ranked_indices = rng.permutation(n_features)
            
        else:
            raise ValueError(f"Unknown feature_selection_method: "
                           f"{self.feature_selection_method}")
        
        return ranked_indices


class MetaAwareCascadeClassifier(ClassifierMixin, BaseEstimator):
    """
    Meta-Aware Cascade Classifier with self-assessment mechanism.
    
    At each cascade level, this variant trains TWO models:
    1. Main classifier/regressor: Predicts the original target (y)
    2. Meta-classifier: Predicts whether the main model will be correct
    
    The meta-classifier is trained on the same features but with a binary target:
    - y_meta = 1 if main model predicted correctly
    - y_meta = 0 if main model predicted incorrectly
    
    For the next cascade level, we select samples that the meta-classifier
    predicts will be classified INCORRECTLY (where the main model
    is likely to fail). This focuses computational resources on genuinely
    difficult samples.
    
    When using regressors as base models, their continuous output is normalized
    to [0,1] and used as decision scores for binary classification.
    
    This is different from standard cascades in several ways:
    - Classical cascade: Passes samples in non-commitment region (uncertain scores)
    - Weighted cascade: Reweights samples in non-commitment region
    - Meta-aware cascade: Passes samples predicted to be misclassified
    
    The meta-classifier provides explicit self-awareness - each level "knows"
    which samples it will likely get wrong and passes them forward.
    
    Parameters
    ----------
    base_classifiers : list of estimators
        Main classifiers or regressors for each cascade level (predict target y).
        - Classifiers must implement `fit` and `predict_proba` or `decision_function`.
        - Regressors must implement `fit` and `predict`.
        
    meta_classifiers : list of estimators, optional
        Meta-classifiers for each level (predict correctness).
        If None, uses the same type as base_classifiers.
        Should have one fewer element than base_classifiers (no meta for last level).
        
    meta_threshold : float, default=0.5
        Threshold for meta-classifier to determine which samples to pass forward.
        Samples with P(correct) < meta_threshold are passed to next level.
        
    non_commitment_strategy : NonCommitmentStrategy or list, default=None
        Strategy for determining non-commitment regions (still used for
        deciding when to commit at prediction time).
        
    threshold : float, 'auto', or 'youden', default='auto'
        Classification threshold for main classifiers.
        
    min_samples_for_next : int, default=10
        Minimum absolute number of samples required to train next classifier.
        
    min_samples_percentage : float, default=0.05
        Minimum percentage (0-1) of original training samples required.
        
    return_noncommitment : bool, default=False
        If True, can return non-commitment label.
        
    noncommitment_label : int, default=-1
        Label for non-commitment decisions.
        
    random_state : int, default=None
        Random seed for reproducibility.
        
    Attributes
    ----------
    cascade_ : list of estimators
        Fitted main classifiers.
        
    meta_cascade_ : list of estimators
        Fitted meta-classifiers (one fewer than cascade_).
        
    meta_thresholds_ : list of float
        Thresholds used for meta-classifiers.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> cascade = MetaAwareCascadeClassifier(
    ...     base_classifiers=[
    ...         RandomForestClassifier(n_estimators=25, random_state=42),
    ...         RandomForestClassifier(n_estimators=50, random_state=42),
    ...         RandomForestClassifier(n_estimators=100, random_state=42)
    ...     ],
    ...     meta_threshold=0.5,
    ...     threshold='youden'
    ... )
    >>> cascade.fit(X_train, y_train)
    >>> predictions = cascade.predict(X_test)
    """
    
    def __init__(self,
                 base_classifiers: List[BaseEstimator],
                 meta_classifiers: Optional[List[BaseEstimator]] = None,
                 meta_threshold: float = 0.5,
                 threshold: Union[float, str] = 'auto',
                 min_samples_for_next: int = 10,
                 min_samples_percentage: float = 0.05,
                 return_noncommitment: bool = False,
                 noncommitment_label: int = -1,
                 random_state: Optional[int] = None):
        
        self.base_classifiers=base_classifiers
        self.threshold=threshold
        self.min_samples_for_next=min_samples_for_next
        self.min_samples_percentage=min_samples_percentage
        self.return_noncommitment=return_noncommitment
        self.noncommitment_label=noncommitment_label
        self.random_state=random_state
        self.meta_classifiers = meta_classifiers
        self.meta_threshold = meta_threshold
    
    def fit(self, X, y):
        """
        Fit the meta-aware cascade classifier.
        
        At each level:
        1. Train main classifier on current dataset
        2. Train meta-classifier to predict correctness
        3. Select samples predicted to be incorrect for next level
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
            
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Check for binary classification
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("MetaAwareCascadeClassifier only supports binary classification")
        
        if not (set(self.classes_) == {0, 1}):
            raise ValueError("Class labels must be 0 and 1")
        
        if self.meta_classifiers is None:
            self.meta_classifiers = [clone(clf) for clf in self.base_classifiers[:-1]]
        
        if len(self.meta_classifiers) != len(self.base_classifiers) - 1:
            raise ValueError("meta_classifiers should have one fewer element than base_classifiers "
                           "(no meta-classifier needed for the last level)")
        
        
        self.cascade_ = []
        self.meta_cascade_ = []
        self.thresholds_ = []
        self.meta_thresholds_ = []
        self.non_commitment_regions_ = []
        
        n_original_samples = len(X)
        min_samples_threshold = max(
            self.min_samples_for_next,
            int(n_original_samples * self.min_samples_percentage)
        )
        
        X_current = X.copy()
        y_current = y.copy()
        
        for i in range(len(self.base_classifiers)):
            n_current_samples = len(X_current)
            percentage_remaining = 100.0 * n_current_samples / n_original_samples
            
            if n_current_samples < min_samples_threshold:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {n_current_samples} samples "
                    f"remaining ({percentage_remaining:.1f}% of original {n_original_samples}), "
                    f"below threshold of {min_samples_threshold} samples"
                )
                break
            
            unique_classes = np.unique(y_current)
            if len(unique_classes) < 2:
                warnings.warn(
                    f"Stopping cascade at level {i}: only {len(unique_classes)} class "
                    f"remaining. Need at least 2 classes."
                )
                break
            
            base_clf = clone(self.base_classifiers[i])
            if self.random_state is not None and hasattr(base_clf, 'random_state'):
                base_clf.random_state = self.random_state + i
            
            base_clf.fit(X_current, y_current)
            
            # Get scores from main classifier
            y_scores = self._get_scores(base_clf, X_current)
            
            # Calculate Youden threshold for main classifier
            if self.threshold == 'youden':
                threshold = calculate_youden_threshold(y_current, y_scores)
            elif self.threshold == 'auto':
                threshold = 0.5
            else:
                threshold = self.threshold
            
            # Use threshold to get predictions (not .predict())
            y_pred_train = (y_scores >= threshold).astype(int)
            
            # Store main classifier and threshold
            self.cascade_.append(base_clf)
            self.thresholds_.append(threshold)
            
            if i < len(self.base_classifiers) - 1:
                y_meta = (y_pred_train == y_current).astype(int)
                
                if len(np.unique(y_meta)) < 2:
                    warnings.warn(
                        f"Stopping cascade at level {i+1}: meta-classifier has only one class "
                        f"(main classifier is {'always correct' if y_meta[0] == 1 else 'always wrong'})"
                    )
                    break
                
                meta_clf = clone(self.meta_classifiers[i])
                if self.random_state is not None and hasattr(meta_clf, 'random_state'):
                    meta_clf.random_state = self.random_state + 1000 + i  # Different seed
                
                meta_clf.fit(X_current, y_meta)
                
                # Get meta-predictions (probability of being correct)
                # Support both classifiers and regressors for meta-models
                if hasattr(meta_clf, 'predict_proba'):
                    meta_proba = meta_clf.predict_proba(X_current)
                    # Probability of being CORRECT (class 1)
                    prob_correct = meta_proba[:, 1]
                elif hasattr(meta_clf, 'decision_function'):
                    # Classifier with decision function
                    scores = meta_clf.decision_function(X_current)
                    # Normalize to [0, 1]
                    prob_correct = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                else:
                    # Assume it's a regressor - use predict() output
                    scores = meta_clf.predict(X_current)
                    scores = np.asarray(scores).ravel()
                    # Normalize to [0, 1]
                    score_min, score_max = scores.min(), scores.max()
                    if score_max - score_min > 1e-10:
                        prob_correct = (scores - score_min) / (score_max - score_min)
                    else:
                        prob_correct = np.full_like(scores, 0.5)
                
                # Calculate Youden threshold for meta-classifier
                meta_threshold_youden = calculate_youden_threshold(y_meta, prob_correct)
                
                # Diagnostic: Check meta-classifier performance on training data
                y_meta_pred = (prob_correct >= meta_threshold_youden).astype(int)
                meta_accuracy = np.mean(y_meta_pred == y_meta)
                
                # Calculate F1 for meta-classifier
                tp = np.sum((y_meta == 1) & (y_meta_pred == 1))
                fp = np.sum((y_meta == 0) & (y_meta_pred == 1))
                fn = np.sum((y_meta == 1) & (y_meta_pred == 0))
                meta_f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
                
                print(f"  Level {i} - Meta-classifier: Accuracy={meta_accuracy:.4f}, F1={meta_f1:.4f}, Threshold={meta_threshold_youden:.4f}")
                
                self.meta_cascade_.append(meta_clf)
                self.meta_thresholds_.append(meta_threshold_youden)
                
                # STEP 3: Select samples for next level
                # Pass samples where meta-classifier predicts INCORRECT classification
                # (prob_correct < meta_threshold means likely to be wrong)
                likely_incorrect = prob_correct < meta_threshold_youden
                
                if not np.any(likely_incorrect):
                    warnings.warn(
                        f"Stopping cascade at level {i+1}: no samples predicted to be incorrect "
                        f"by meta-classifier"
                    )
                    break
                
                # Update data for next level
                X_current = X_current[likely_incorrect]
                y_current = y_current[likely_incorrect]
            
        self.n_cascade_levels_ = len(self.cascade_)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using the meta-aware cascade.
        
        At prediction time, we use the meta-classifiers to determine when to pass
        samples to the next level. Samples where the meta-classifier predicts
        P(correct) < meta_threshold are passed forward to the next classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, np.nan)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold) in enumerate(zip(self.cascade_, self.thresholds_)):
            # Only process uncommitted samples
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            scores = self._get_scores(clf, X_remaining)
            
            # Get predictions from main classifier
            y_pred_level = (scores >= threshold).astype(int)
            
            # If this is not the last level and we have a meta-classifier, use it
            if i < len(self.meta_cascade_):
                meta_clf = self.meta_cascade_[i]
                
                # Get meta-classifier predictions (probability of being correct)
                if hasattr(meta_clf, 'predict_proba'):
                    meta_proba = meta_clf.predict_proba(X_remaining)
                    prob_correct = meta_proba[:, 1]
                elif hasattr(meta_clf, 'decision_function'):
                    # SVM-like classifiers
                    meta_scores = meta_clf.decision_function(X_remaining)
                    prob_correct = (meta_scores - meta_scores.min()) / (meta_scores.max() - meta_scores.min() + 1e-10)
                else:
                    # Regressors: use predict and normalize
                    meta_preds = meta_clf.predict(X_remaining)
                    prob_correct = (meta_preds - meta_preds.min()) / (meta_preds.max() - meta_preds.min() + 1e-10)
                
                # Commit samples where meta-classifier is confident (prob_correct >= threshold)
                confident_mask = prob_correct >= self.meta_thresholds_[i]
                confident_idx = np.where(mask)[0][confident_mask]
                
                # Assign predictions for confident samples
                predictions[confident_idx] = y_pred_level[confident_mask]
                committed[confident_idx] = True
                
                # Uncertain samples (prob_correct < threshold) will be passed to next level
            else:
                # Last level: commit all remaining samples
                remaining_idx = np.where(mask)[0]
                predictions[remaining_idx] = y_pred_level
                committed[remaining_idx] = True
        
        # Handle remaining non-committed samples (shouldn't happen if cascade is complete)
        if self.return_noncommitment:
            predictions[~committed] = self.noncommitment_label
        else:
            if np.any(~committed):
                X_remaining = X[~committed]
                scores = self._get_scores(self.cascade_[-1], X_remaining)
                predictions[~committed] = (scores >= self.thresholds_[-1]).astype(int)
        
        return predictions.astype(int)
    
    def predict_with_meta_confidence(self, X):
        """
        Predict class labels along with meta-classifier confidence.
        
        Returns predictions and the meta-classifier's estimated probability
        that each prediction is correct.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
            
        confidence : ndarray of shape (n_samples,)
            Meta-classifier's estimated probability of correctness.
            For samples decided at level i, uses meta_cascade_[i] if available.
            
        depths : ndarray of shape (n_samples,)
            Cascade level at which each prediction was made.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, np.nan)
        confidence = np.full(n_samples, np.nan)
        depths = np.full(n_samples, self.n_cascade_levels_, dtype=int)
        committed = np.zeros(n_samples, dtype=bool)
        
        for i, (clf, threshold) in enumerate(zip(self.cascade_, self.thresholds_)):
            mask = ~committed
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            scores = self._get_scores(clf, X_remaining)
            
            # Get predictions from main classifier
            y_pred_level = (scores >= threshold).astype(int)
            
            # If this is not the last level and we have a meta-classifier, use it
            if i < len(self.meta_cascade_):
                meta_clf = self.meta_cascade_[i]
                
                # Get meta-classifier predictions (probability of being correct)
                # Support both classifiers and regressors for meta-models
                if hasattr(meta_clf, 'predict_proba'):
                    meta_proba = meta_clf.predict_proba(X_remaining)
                    prob_correct = meta_proba[:, 1]
                elif hasattr(meta_clf, 'decision_function'):
                    # Classifier with decision function
                    meta_scores = meta_clf.decision_function(X_remaining)
                    prob_correct = (meta_scores - meta_scores.min()) / (meta_scores.max() - meta_scores.min() + 1e-10)
                else:
                    # Assume it's a regressor - use predict() output
                    meta_scores = meta_clf.predict(X_remaining)
                    meta_scores = np.asarray(meta_scores).ravel()
                    score_min, score_max = meta_scores.min(), meta_scores.max()
                    if score_max - score_min > 1e-10:
                        prob_correct = (meta_scores - score_min) / (score_max - score_min)
                    else:
                        prob_correct = np.full_like(meta_scores, 0.5)
                
                # Commit samples where meta-classifier is confident
                confident_mask = prob_correct >= self.meta_thresholds_[i]
                confident_idx = np.where(mask)[0][confident_mask]
                
                if len(confident_idx) > 0:
                    predictions[confident_idx] = y_pred_level[confident_mask]
                    confidence[confident_idx] = prob_correct[confident_mask]
                    depths[confident_idx] = i
                    committed[confident_idx] = True
            else:
                # Last level: commit all remaining samples
                remaining_idx = np.where(mask)[0]
                predictions[remaining_idx] = y_pred_level
                confidence[remaining_idx] = np.nan  # No meta-classifier for last level
                depths[remaining_idx] = i
                committed[remaining_idx] = True
        
        # Handle non-committed samples (shouldn't happen if cascade is complete)
        if self.return_noncommitment:
            predictions[~committed] = self.noncommitment_label
            confidence[~committed] = 0.0  # Low confidence for non-committed
        else:
            if np.any(~committed):
                X_remaining = X[~committed]
                scores = self._get_scores(self.cascade_[-1], X_remaining)
                predictions[~committed] = (scores >= self.thresholds_[-1]).astype(int)
                confidence[~committed] = np.nan  # No meta-classifier for last level
        
        return predictions.astype(int), confidence, depths

    def _get_scores(self, clf, X):
        """
        Get decision scores from a classifier or regressor.
        
        For classifiers: tries predict_proba first, then decision_function.
        For regressors: uses predict() output as scores, normalized to [0,1].
        """
        if hasattr(clf, 'predict_proba'):
            probas = clf.predict_proba(X)
            if probas.shape[1] == 2:
                return probas[:, 1]
            else:
                return probas[:, -1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X)
            def normalize(results):
                res = (results - results.min()) / (results.max() - results.min() + 1e-10)
                return res
            return normalize(scores)
        else:
            # Assume it's a regressor - use predict() output as scores
            scores = clf.predict(X)
            scores = np.asarray(scores).ravel()
            # Normalize to [0, 1]
            score_min, score_max = scores.min(), scores.max()
            if score_max - score_min > 1e-10:
                scores = (scores - score_min) / (score_max - score_min)
            else:
                # All scores are the same, return 0.5
                scores = np.full_like(scores, 0.5)
            return scores

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        flip_y=0.3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("=" * 70)
    print("Three-Way Decision Cascade Classifier Demo")
    print("=" * 70)
    
    cascade = ThreeWayDecisionCascadeClassifier(
        base_classifiers=[
            RandomForestClassifier(n_estimators=2, random_state=42),
            LogisticRegression(max_iter=1000, random_state=42),
            RandomForestClassifier(n_estimators=3, random_state=43)
        ],
        non_commitment_strategy=AdaptiveF1Strategy(penalization=1.0),
        threshold='youden',
        min_samples_for_next=20,
        min_samples_percentage=0.15
    )
    
    print("\nTraining cascade...")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Automatic truncation threshold: {int(len(X_train) * 0.15)} samples (15%)")
    cascade.fit(X_train, y_train)
    
    print(f"\nCascade trained with {cascade.n_cascade_levels_} levels")
    print("\nNon-commitment regions:")
    for i, (threshold, region) in enumerate(
        zip(cascade.thresholds_, cascade.non_commitment_regions_)
    ):
        print(f"  Level {i}: threshold={threshold:.4f}, "
              f"region=[{region[0]:.4f}, {region[1]:.4f}]")
    
    y_pred = cascade.predict(X_test)
    y_pred_with_depth, depths = cascade.predict_with_depth(X_test)
    
    print("\n" + "=" * 70)
    print("Results on Test Set")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 70)
    print("Cascade Statistics")
    print("=" * 70)
    stats = cascade.get_cascade_statistics(X_test, y_test)
    for stat in stats:
        print(f"\nLevel {stat['level']}:")
        print(f"  Samples decided: {stat['n_decided']}")
        print(f"  Samples remaining: {stat['n_remaining']}")
        print(f"  Accuracy: {stat['accuracy']:.4f}")
        print(f"  Threshold: {stat['threshold']:.4f}")
        print(f"  Region: [{stat['region'][0]:.4f}, {stat['region'][1]:.4f}]")
    
    print("\n" + "=" * 70)
    print("Depth Distribution")
    print("=" * 70)
    for i in range(cascade.n_cascade_levels_):
        n_at_depth = np.sum(depths == i)
        pct = 100 * n_at_depth / len(depths)
        print(f"Level {i}: {n_at_depth} samples ({pct:.1f}%)")
    
    n_final = np.sum(depths == cascade.n_cascade_levels_)
    pct_final = 100 * n_final / len(depths)
    print(f"Final (non-committed): {n_final} samples ({pct_final:.1f}%)")
    
    # Demo: Progressive Feature Cascade
    print("\n" + "=" * 70)
    print("Progressive Feature Cascade Classifier Demo")
    print("=" * 70)
    
    X_prog, y_prog = make_classification(
        n_samples=3000,
        n_features=50,  # More features to demonstrate progressive selection
        n_informative=40,
        n_redundant=5,
        n_repeated=5,
        flip_y=0.2,
        random_state=42
    )
    
    X_train_prog, X_test_prog, y_train_prog, y_test_prog = train_test_split(
        X_prog, y_prog, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: {X_prog.shape[0]} samples, {X_prog.shape[1]} features")
    print(f"Training set: {len(X_train_prog)} samples")
    print(f"Test set: {len(X_test_prog)} samples")
    
    prog_cascade = ProgressiveFeatureCascadeClassifier(
        base_classifiers=[
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(n_estimators=20),
            RandomForestClassifier(n_estimators=50)
        ],
        feature_percentages=[0.2, 0.4, 1.0],
        feature_selection_method='variance',
        non_commitment_strategy=AdaptiveF1Strategy(penalization=0.0),
        threshold='youden',
        min_samples_for_next=20,
        random_state=42
    )
    
    print("\nTraining progressive feature cascade...")
    print("Feature progression: 20% → 40% → 100%")
    prog_cascade.fit(X_train_prog, y_train_prog)
    
    print(f"\nCascade trained with {prog_cascade.n_cascade_levels_} levels")
    print("\nFeature usage per level:")
    for i, (n_feat, pct) in enumerate(
        zip([len(idx) for idx in prog_cascade.feature_indices_[:prog_cascade.n_cascade_levels_]],
            prog_cascade.feature_percentages_used_[:prog_cascade.n_cascade_levels_])
    ):
        print(f"  Level {i}: {n_feat} features ({pct*100:.0f}%)")
    
    y_pred_prog = prog_cascade.predict(X_test_prog)
    y_pred_prog_depth, depths_prog = prog_cascade.predict_with_depth(X_test_prog)
    
    print("\n" + "=" * 70)
    print("Results on Test Set")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy_score(y_test_prog, y_pred_prog):.4f}")
    
    print("\n" + "=" * 70)
    print("Progressive Cascade Statistics")
    print("=" * 70)
    stats_prog = prog_cascade.get_cascade_statistics(X_test_prog, y_test_prog)
    for stat in stats_prog:
        print(f"\nLevel {stat['level']}:")
        print(f"  Features used: {stat['n_features']} ({stat['feature_percentage']*100:.0f}%)")
        print(f"  Samples decided: {stat['n_decided']}")
        print(f"  Samples remaining: {stat['n_remaining']}")
        print(f"  Accuracy: {stat['accuracy']:.4f}")
    
    print("\n" + "=" * 70)
    print("Depth Distribution")
    print("=" * 70)
    for i in range(prog_cascade.n_cascade_levels_):
        n_at_depth = np.sum(depths_prog == i)
        pct = 100 * n_at_depth / len(depths_prog)
        print(f"Level {i}: {n_at_depth} samples ({pct:.1f}%)")
    
    n_final_prog = np.sum(depths_prog == prog_cascade.n_cascade_levels_)
    pct_final_prog = 100 * n_final_prog / len(depths_prog)
    print(f"Final (non-committed): {n_final_prog} samples ({pct_final_prog:.1f}%)")
    
    # Demo: Meta-Aware Cascade
    print("\n" + "=" * 70)
    print("Meta-Aware Cascade Classifier Demo")
    print("=" * 70)
    
    X_meta, y_meta = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_repeated=0,
        n_clusters_per_class=3,
        flip_y=0.15,  # Moderate noise
        class_sep=0.8,  # Moderate separation
        random_state=42
    )
    
    X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
        X_meta, y_meta, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset: {X_meta.shape[0]} samples, {X_meta.shape[1]} features")
    print(f"Training set: {len(X_train_meta)} samples")
    print(f"Test set: {len(X_test_meta)} samples")
    print(f"Class distribution: {np.bincount(y_train_meta)}")
    
    meta_cascade = MetaAwareCascadeClassifier(
        base_classifiers=[
            LogisticRegression(max_iter=1000, random_state=42),  # Simple model first
            RandomForestClassifier(n_estimators=20, max_depth=5, random_state=43),  # Medium complexity
            RandomForestClassifier(n_estimators=50, random_state=44),  # Complex model
            RandomForestClassifier(n_estimators=100, random_state=45)  # Final complex model
        ],
        meta_classifiers=None,  # Will use clones of base classifiers
        threshold='youden',
        min_samples_for_next=30,
        min_samples_percentage=0.03,  # Lower threshold to allow more levels
        random_state=42
    )
    
    print("\nTraining meta-aware cascade...")
    print("Each level trains:")
    print("  1. Main classifier (predicts target class)")
    print("  2. Meta-classifier (predicts if main classifier will be correct)")
    meta_cascade.fit(X_train_meta, y_train_meta)
    
    print(f"\nCascade trained with {meta_cascade.n_cascade_levels_} levels")
    print(f"Meta-classifiers trained: {len(meta_cascade.meta_cascade_)}")
    
    print("\nMeta-classifier thresholds (Youden):")
    for i, meta_thresh in enumerate(meta_cascade.meta_thresholds_):
        print(f"  Level {i}: {meta_thresh:.4f}")
    
    y_pred_meta = meta_cascade.predict(X_test_meta)
    y_pred_meta_conf, confidence, depths_meta = meta_cascade.predict_with_meta_confidence(X_test_meta)
    
    print("\n" + "=" * 70)
    print("Results on Test Set")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy_score(y_test_meta, y_pred_meta):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_meta, y_pred_meta))
    
    print("\n" + "=" * 70)
    print("Meta-Classifier Confidence Analysis")
    print("=" * 70)
    for i in range(meta_cascade.n_cascade_levels_):
        mask = depths_meta == i
        if np.any(mask):
            n_at_level = np.sum(mask)
            pct = 100 * n_at_level / len(depths_meta)
            
            # Get confidence for this level (only for levels with meta-classifier)
            if i < len(meta_cascade.meta_cascade_):
                conf_at_level = confidence[mask]
                valid_conf = conf_at_level[~np.isnan(conf_at_level)]
                if len(valid_conf) > 0:
                    print(f"\nLevel {i}: {n_at_level} samples ({pct:.1f}%)")
                    print(f"  Mean confidence: {valid_conf.mean():.4f}")
                    print(f"  Min confidence:  {valid_conf.min():.4f}")
                    print(f"  Max confidence:  {valid_conf.max():.4f}")
                    
                    # Accuracy for samples decided at this level
                    acc_at_level = accuracy_score(y_test_meta[mask], y_pred_meta[mask])
                    print(f"  Accuracy: {acc_at_level:.4f}")
            else:
                print(f"\nLevel {i} (final): {n_at_level} samples ({pct:.1f}%)")
                acc_at_level = accuracy_score(y_test_meta[mask], y_pred_meta[mask])
                print(f"  Accuracy: {acc_at_level:.4f}")
    
    print("\n" + "=" * 70)
    print("Depth Distribution")
    print("=" * 70)
    for i in range(meta_cascade.n_cascade_levels_):
        n_at_depth = np.sum(depths_meta == i)
        pct = 100 * n_at_depth / len(depths_meta)
        print(f"Level {i}: {n_at_depth} samples ({pct:.1f}%)")


class ComparativeMetaCascadeClassifier(BaseEstimator, ClassifierMixin):
    """
    Cascade classifier with comparative meta-classifiers.
    
    Unlike traditional cascades, all base classifiers are trained on the same data.
    Meta-classifiers are trained to predict whether the next classifier will perform
    better than the current one, enabling intelligent routing decisions.
    
    Architecture:
    1. Train all base classifiers on the full training set (no cascade structure)
    2. For each level i, train a meta-classifier to predict: 
       y_meta = 1 if classifier[i+1] predicts correctly AND classifier[i] predicts incorrectly
       y_meta = 0 otherwise
    3. During prediction, use meta-classifiers to decide whether to continue to next level
    
    Parameters
    ----------
    base_classifiers : list of classifiers or regressors
        Base models for each cascade level. All trained on same data.
    meta_classifiers : list of classifiers or regressors
        Meta-models to predict comparative performance (length = len(base_classifiers) - 1)
    threshold_method : str, default='youden'
        Method for computing thresholds ('fixed', 'youden', 'percentile')
    meta_threshold_method : str, default='youden'
        Method for computing meta-classifier thresholds
    fixed_threshold : float, default=0.5
        Threshold value when using 'fixed' method
    percentile_threshold : float, default=90
        Percentile value when using 'percentile' method
    min_samples_per_level : int, default=20
        Minimum samples to continue cascade
    random_state : int, RandomState instance or None, default=None
        Controls randomness for reproducible results
    verbose : bool, default=True
        Print training progress
    """
    
    def __init__(self, base_classifiers, meta_classifiers,
                 threshold_method='youden', meta_threshold_method='youden',
                 fixed_threshold=0.5, percentile_threshold=90,
                 min_samples_per_level=20, random_state=None, verbose=True):
        self.base_classifiers = base_classifiers
        self.meta_classifiers = meta_classifiers
        self.threshold_method = threshold_method
        self.meta_threshold_method = meta_threshold_method
        self.fixed_threshold = fixed_threshold
        self.percentile_threshold = percentile_threshold
        self.min_samples_per_level = min_samples_per_level
        self.random_state = random_state
        self.verbose = verbose
        
    def _get_scores(self, clf, X):
        """Get decision scores from classifier or regressor."""
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X)
            return proba[:, 1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            # Regressor
            preds = clf.predict(X)
            return (preds - preds.min()) / (preds.max() - preds.min() + 1e-10)
    
    def _compute_threshold(self, scores, y_true, method='youden'):
        """Compute optimal threshold using specified method."""
        if method == 'fixed':
            return self.fixed_threshold
        elif method == 'percentile':
            return np.percentile(scores, self.percentile_threshold)
        elif method == 'youden':
            return calculate_youden_threshold(y_true, scores)
        else:
            return 0.5
    
    def fit(self, X, y):
        """
        Fit the comparative meta cascade.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        
        n_levels = len(self.base_classifiers)
        
        if len(self.meta_classifiers) != n_levels - 1:
            raise ValueError(
                f"Need {n_levels - 1} meta-classifiers for {n_levels} base classifiers, "
                f"got {len(self.meta_classifiers)}"
            )
        
        self.cascade_ = []
        self.thresholds_ = []
        self.meta_cascade_ = []
        self.meta_thresholds_ = []
        
        # Step 1: Train all base classifiers on the same data
        if self.verbose:
            print(f"\nTraining {n_levels} base classifiers on full dataset ({len(X)} samples)...")
        
        all_predictions = []
        all_scores = []
        
        for i, clf in enumerate(self.base_classifiers):
            clf_copy = clone(clf)
            if self.random_state is not None and hasattr(clf_copy, 'random_state'):
                clf_copy.set_params(random_state=self.random_state + i)
            clf_copy.fit(X, y)
            self.cascade_.append(clf_copy)
            
            # Get predictions and scores
            scores = self._get_scores(clf_copy, X)
            threshold = self._compute_threshold(scores, y, self.threshold_method)
            self.thresholds_.append(threshold)
            
            y_pred = (scores >= threshold).astype(int)
            all_predictions.append(y_pred)
            all_scores.append(scores)
            
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='binary', zero_division=0)
            
            if self.verbose:
                print(f"  Level {i} - Base classifier: Accuracy={acc:.4f}, F1={f1:.4f}, Threshold={threshold:.4f}")
        
        # Step 2: Train meta-classifiers to predict comparative performance
        if self.verbose:
            print(f"\nTraining {len(self.meta_classifiers)} meta-classifiers...")
        
        for i in range(n_levels - 1):
            # Create meta-training data:
            # y_meta = 1 if next classifier improves (correct when current is wrong)
            # y_meta = 0 otherwise
            current_correct = (all_predictions[i] == y)
            next_correct = (all_predictions[i + 1] == y)
            
            # Improvement: next is correct AND current is wrong
            y_meta = (next_correct & ~current_correct).astype(int)
            
            meta_clf = clone(self.meta_classifiers[i])
            if self.random_state is not None and hasattr(meta_clf, 'random_state'):
                meta_clf.set_params(random_state=self.random_state + 1000 + i)
            meta_clf.fit(X, y_meta)
            self.meta_cascade_.append(meta_clf)
            
            # Get meta-classifier scores and threshold
            if hasattr(meta_clf, 'predict_proba'):
                meta_proba = meta_clf.predict_proba(X)
                meta_scores = meta_proba[:, 1]
            elif hasattr(meta_clf, 'decision_function'):
                raw_scores = meta_clf.decision_function(X)
                meta_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
            else:
                # Regressor
                raw_preds = meta_clf.predict(X)
                meta_scores = (raw_preds - raw_preds.min()) / (raw_preds.max() - raw_preds.min() + 1e-10)
            
            meta_threshold = self._compute_threshold(meta_scores, y_meta, self.meta_threshold_method)
            self.meta_thresholds_.append(meta_threshold)
            
            # Evaluate meta-classifier
            y_meta_pred = (meta_scores >= meta_threshold).astype(int)
            meta_acc = accuracy_score(y_meta, y_meta_pred)
            meta_f1 = f1_score(y_meta, y_meta_pred, average='binary', zero_division=0)
            
            improvement_rate = y_meta.mean()
            
            if self.verbose:
                print(f"  Level {i} - Meta-classifier: Accuracy={meta_acc:.4f}, F1={meta_f1:.4f}, "
                      f"Threshold={meta_threshold:.4f}, ImprovementRate={improvement_rate:.4f}")
        
        self.n_cascade_levels_ = n_levels
        self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        """
        Predict using the comparative meta cascade.
        
        Samples flow through the cascade based on meta-classifier predictions.
        If meta-classifier predicts improvement (prob >= threshold), continue to next level.
        Otherwise, commit with current classifier's prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        n_samples = len(X)
        
        predictions = np.zeros(n_samples, dtype=int)
        committed = np.zeros(n_samples, dtype=bool)
        decision_levels = np.zeros(n_samples, dtype=int)
        
        # Start with all samples at level 0
        X_remaining = X.copy()
        mask = ~committed
        
        for i in range(self.n_cascade_levels_):
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            clf = self.cascade_[i]
            threshold = self.thresholds_[i]
            
            # Get predictions from current classifier
            scores = self._get_scores(clf, X_remaining)
            y_pred_level = (scores >= threshold).astype(int)
            
            # Decide whether to commit or continue
            if i < len(self.meta_cascade_):
                # Use meta-classifier to predict if next level will improve
                meta_clf = self.meta_cascade_[i]
                
                # Get meta-classifier predictions
                if hasattr(meta_clf, 'predict_proba'):
                    meta_proba = meta_clf.predict_proba(X_remaining)
                    prob_improvement = meta_proba[:, 1]
                elif hasattr(meta_clf, 'decision_function'):
                    meta_scores = meta_clf.decision_function(X_remaining)
                    prob_improvement = (meta_scores - meta_scores.min()) / (meta_scores.max() - meta_scores.min() + 1e-10)
                else:
                    # Regressor
                    meta_preds = meta_clf.predict(X_remaining)
                    prob_improvement = (meta_preds - meta_preds.min()) / (meta_preds.max() - meta_preds.min() + 1e-10)
                
                # Commit samples where improvement is unlikely
                no_improvement_mask = prob_improvement < self.meta_thresholds_[i]
                commit_idx = np.where(mask)[0][no_improvement_mask]
                
                predictions[commit_idx] = y_pred_level[no_improvement_mask]
                committed[commit_idx] = True
                decision_levels[commit_idx] = i
                
                # Samples with predicted improvement continue to next level
                mask = ~committed
            else:
                # Last level: commit all remaining samples
                remaining_idx = np.where(mask)[0]
                predictions[remaining_idx] = y_pred_level
                committed[remaining_idx] = True
                decision_levels[remaining_idx] = i
                mask = ~committed
        
        return predictions
    
    def predict_with_depths(self, X):
        """
        Predict and return decision depths.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        depths : array of shape (n_samples,)
            Decision level for each sample
        """
        X = np.array(X)
        n_samples = len(X)
        
        predictions = np.zeros(n_samples, dtype=int)
        committed = np.zeros(n_samples, dtype=bool)
        decision_levels = np.zeros(n_samples, dtype=int)
        
        mask = ~committed
        
        for i in range(self.n_cascade_levels_):
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            clf = self.cascade_[i]
            threshold = self.thresholds_[i]
            
            scores = self._get_scores(clf, X_remaining)
            y_pred_level = (scores >= threshold).astype(int)
            
            if i < len(self.meta_cascade_):
                meta_clf = self.meta_cascade_[i]
                
                if hasattr(meta_clf, 'predict_proba'):
                    meta_proba = meta_clf.predict_proba(X_remaining)
                    prob_improvement = meta_proba[:, 1]
                elif hasattr(meta_clf, 'decision_function'):
                    meta_scores = meta_clf.decision_function(X_remaining)
                    prob_improvement = (meta_scores - meta_scores.min()) / (meta_scores.max() - meta_scores.min() + 1e-10)
                else:
                    meta_preds = meta_clf.predict(X_remaining)
                    prob_improvement = (meta_preds - meta_preds.min()) / (meta_preds.max() - meta_preds.min() + 1e-10)
                
                no_improvement_mask = prob_improvement < self.meta_thresholds_[i]
                commit_idx = np.where(mask)[0][no_improvement_mask]
                
                predictions[commit_idx] = y_pred_level[no_improvement_mask]
                committed[commit_idx] = True
                decision_levels[commit_idx] = i
                mask = ~committed
            else:
                remaining_idx = np.where(mask)[0]
                predictions[remaining_idx] = y_pred_level
                committed[remaining_idx] = True
                decision_levels[remaining_idx] = i
                mask = ~committed
        
        return predictions, decision_levels
    
    def predict_with_meta_confidence(self, X):
        """
        Predict class labels along with meta-classifier confidence scores.
        
        This method is provided for compatibility with evaluation frameworks.
        The confidence scores represent the meta-classifier's probability that
        the next level will NOT improve (i.e., 1 - prob_improvement).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        confidence : array of shape (n_samples,)
            Confidence scores (1 - prob_improvement for committed samples)
        depths : array of shape (n_samples,)
            Decision level for each sample
        """
        X = np.array(X)
        n_samples = len(X)
        
        predictions = np.zeros(n_samples, dtype=int)
        confidence = np.ones(n_samples, dtype=float)  # Default high confidence
        committed = np.zeros(n_samples, dtype=bool)
        decision_levels = np.zeros(n_samples, dtype=int)
        
        mask = ~committed
        
        for i in range(self.n_cascade_levels_):
            if not np.any(mask):
                break
            
            X_remaining = X[mask]
            clf = self.cascade_[i]
            threshold = self.thresholds_[i]
            
            scores = self._get_scores(clf, X_remaining)
            y_pred_level = (scores >= threshold).astype(int)
            
            if i < len(self.meta_cascade_):
                meta_clf = self.meta_cascade_[i]
                
                if hasattr(meta_clf, 'predict_proba'):
                    meta_proba = meta_clf.predict_proba(X_remaining)
                    prob_improvement = meta_proba[:, 1]
                elif hasattr(meta_clf, 'decision_function'):
                    meta_scores = meta_clf.decision_function(X_remaining)
                    prob_improvement = (meta_scores - meta_scores.min()) / (meta_scores.max() - meta_scores.min() + 1e-10)
                else:
                    meta_preds = meta_clf.predict(X_remaining)
                    prob_improvement = (meta_preds - meta_preds.min()) / (meta_preds.max() - meta_preds.min() + 1e-10)
                
                # Confidence = probability of no improvement (decision to commit)
                conf_scores = 1.0 - prob_improvement
                
                no_improvement_mask = prob_improvement < self.meta_thresholds_[i]
                commit_idx = np.where(mask)[0][no_improvement_mask]
                
                predictions[commit_idx] = y_pred_level[no_improvement_mask]
                confidence[commit_idx] = conf_scores[no_improvement_mask]
                committed[commit_idx] = True
                decision_levels[commit_idx] = i
                mask = ~committed
            else:
                remaining_idx = np.where(mask)[0]
                predictions[remaining_idx] = y_pred_level
                # Last level always has high confidence (no alternative)
                confidence[remaining_idx] = 1.0
                committed[remaining_idx] = True
                decision_levels[remaining_idx] = i
                mask = ~committed
        
        return predictions, confidence, decision_levels
