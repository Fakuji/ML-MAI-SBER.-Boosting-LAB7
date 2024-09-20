from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
            loss_function=mse_loss
    ):
        self.loss_function = loss_function
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators
        self.models: list = []
        self.gammas: list = []
        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)
        self.plot: bool = plot
        self.history = defaultdict(list)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        
        
    def _optimize_gamma(self, X_bootstrap, residuals, model, loss_function):
    # Определение функции потерь для минимизации
      def loss_function_with_gamma(gamma, X, residuals, model, loss_function):
        y_pred = model.predict(X) * gamma
        return loss_function(residuals, y_pred)
    
    # Минимизация одномерной функции потерь для определения оптимальной гаммы
      result = minimize_scalar(
        loss_function_with_gamma, 
        bounds=(0, 1), 
        args=(X_bootstrap, residuals, model, loss_function), 
        method='bounded'
      )

      if result.success:
        return result.x  # Возвращаем оптимизированную гамму
      else:
        raise ValueError("Ой ладно, гамма не хочет сотрудничать, что тут поделаешь...") 
        
        
    def fit_new_base_model(self, X, y, current_preds):
    # Генерируем бутстрап-выборку
        indices = np.random.choice(X.shape[0], int(X.shape[0] * self.subsample), replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        residuals = y_bootstrap - current_preds[indices]
    
    # Обучаем базовую модель
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_bootstrap, residuals)
    
    # Оптимизация гаммы - тут может быть ваш метод
        gamma = self._optimize_gamma(X_bootstrap, residuals, model, self.loss_function)
    
        return model, gamma
        
        
        
        
        
    def fit(self, X_train, y_train, X_val, y_val, loss_fn):
    # Инициализация начальных предсказаний
        train_preds = np.zeros(y_train.shape)
        val_preds = np.zeros(y_val.shape)
        best_loss = np.inf
        best_round = 0
        train_losses = []
        val_losses = []

        for i in range(self.n_estimators):
        # Обучаем новую базовую модель на остатках
            model, gamma = self.fit_new_base_model(X_train, y_train, train_preds)
        
        # Обновляем предсказания
            train_preds += self.learning_rate * gamma * model.predict(X_train)
            val_preds += self.learning_rate * gamma * model.predict(X_val)
        
        # Рассчитываем ошибки
            train_loss = loss_fn(y_train, train_preds)
            val_loss = loss_fn(y_val, val_preds)
        
        # Добавляем модель и гамму в список
            self.models.append(model)
            self.gammas.append(gamma)
        
        # Ранняя остановка
            if val_loss < best_loss:
                best_loss = val_loss
                best_round = i
            elif (i - best_round) >= self.early_stopping_rounds:
                print(f"Stopping early at round {i + 1}")
                break

            if self.plot:
            # Сюда добавим код для графика позже...
                train_losses.append(train_loss)
                val_losses.append(val_loss)

        # Если нужно, отрисуем график
        if self.plot:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Over Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        
        
        
        
        
    def predict_proba(self, X):
        # Суммируем предсказания базовых моделей
      preds = np.sum([gamma * model.predict(X) for gamma, model in zip(self.gammas, self.models)], axis=0)
        # Применяем сигмоидальную функцию
      proba_class_1 = 1 / (1 + np.exp(-preds))
      # Рассчитываем вероятности для класса 0
      proba_class_0 = 1 - proba_class_1
    
      # Объединяем вероятности для классов 0 и 1 в двумерный массив
      return np.vstack((proba_class_0, proba_class_1)).T
            
            
            
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        # Находит оптимальное значение гаммы для минимизации функции потерь.
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]
    def score(self, x, y):
        return score(self, x, y)
    @property
    def feature_importances_(self):
    # Проверяем, используется ли атрибут feature_importances_ в базовой модели
        if not hasattr(self.models[0], 'feature_importances_'):
            raise AttributeError("Base model doesn't have attribute feature_importances_")
        
    # Инициализируем массив для суммирования важностей признаков
        total_importances = np.zeros_like(self.models[0].feature_importances_)
    
    # Суммируем важность каждого признака из всех моделей, учитывая веса (гаммы) моделей
        for gamma, model in zip(self.gammas, self.models):
            total_importances += gamma * model.feature_importances_
    
    # Нормируем важности признаков так, чтобы сумма всех значений была равна 1
        total_importances /= total_importances.sum()

        return total_importances
