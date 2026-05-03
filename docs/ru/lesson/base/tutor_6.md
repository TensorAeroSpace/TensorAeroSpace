# Урок 6 — Наблюдатели состояния и принцип разделения

## 1. Проблема неполной информации

На прошлом семинаре мы научились синтезировать регулятор с помощью метода размещения полюсов. Этот метод требует, чтобы закон управления `u = -Kx` был основан на полном векторе состояния `x`. Однако на практике измерение всех переменных состояния часто бывает:

*   **Технически невозможным:** Некоторые состояния, такие как скорость изменения углов, могут не измеряться напрямую.
*   **Экономически нецелесообразным:** Установка большого количества датчиков усложняет и удорожает систему.

Мы сталкиваемся с проблемой: для управления нам нужен полный вектор состояния `x`, но мы можем измерить только вектор выхода `y = Cx`.

## 2. Идея наблюдателя состояния

Решение этой проблемы – построить **наблюдатель состояния** (state observer) или **оценщик состояния** (state estimator). Это программная или аппаратная модель, которая вычисляет **оценку** вектора состояния, обозначаемую как $\hat{x}$ (x с крышкой).

Идея наблюдателя проста: мы создаем копию нашей реальной системы и "прогоняем" ее параллельно. Затем мы постоянно сравниваем реальный выход системы `y` с выходом нашей модели-наблюдателя `ŷ` и используем разницу (ошибку) для коррекции состояния нашей модели, чтобы оно сходилось к реальному состоянию.

## 3. Наблюдатель Люенбергера

Самым распространенным является **наблюдатель Люенбергера**. Его структура описывается следующим уравнением:

```
✰̇ = A✰ + Bu + L(y - ŷ)
```

где:
*   `✰` – оценка вектора состояния.
*   `A✰ + Bu` – это наша модель реальной системы. Она использует те же матрицы `A` и `B` и тот же входной сигнал `u`.
*   `y` – реальный, измеряемый выход системы.
*   `ŷ = C✰` – оценка выхода, полученная из нашей модели.
*   `(y - ŷ)` – ошибка между реальным и оцененным выходом.
*   `L` – **матрица усиления наблюдателя**, которую нам нужно спроектировать.

Член `L(y - ŷ)` – это и есть корректирующая обратная связь. Если выход нашей модели отличается от реального, этот член "подталкивает" состояние модели `✰` в нужную сторону, чтобы ошибка уменьшалась.

## 4. Динамика ошибки наблюдения

Определим ошибку наблюдения как `e = x - ✰`. Наша цель – сделать так, чтобы эта ошибка стремилась к нулю как можно быстрее. Найдем уравнение, описывающее динамику ошибки:

```
ė = ẋ - ✰̇
ė = (Ax + Bu) - (A✰ + Bu + L(y - ŷ))
ė = Ax - A✰ - L(Cx - C✰)
ė = A(x - ✰) - LC(x - ✰)
ė = (A - LC)e
```

Мы получили автономное линейное уравнение `ė = (A - LC)e`. Динамика ошибки наблюдения определяется исключительно матрицей `(A - LC)` и не зависит от входа `u`.

Устойчивость этой системы (то есть стремление ошибки `e` к нулю) зависит от собственных чисел матрицы `(A - LC)`. Мы можем выбрать матрицу `L` так, чтобы разместить эти собственные числа (полюсы наблюдателя) в желаемых местах комплексной плоскости.

**Теорема:** Если система `(A, C)` полностью наблюдаема, то мы можем разместить полюсы наблюдателя (собственные числа матрицы `A - LC`) в любых желаемых местах, выбрав соответствующую матрицу `L`.

Это дуальный результат по отношению к теореме о размещении полюсов для регулятора. Наблюдаемость является необходимым и достаточным условием для построения наблюдателя с любой желаемой динамикой ошибки.

## 5. Принцип разделения

Теперь у нас есть две отдельные задачи:

1.  **Синтез регулятора:** Найти матрицу `K` для размещения полюсов замкнутой системы `(A - BK)` в желаемых местах (предполагая, что `x` известен).
2.  **Синтез наблюдателя:** Найти матрицу `L` для размещения полюсов наблюдателя `(A - LC)` в желаемых местах, чтобы ошибка `e` быстро затухала.

**Принцип разделения (Separation Principle)** – это фундаментальный результат в теории управления, который гласит, что эти две задачи можно решать **независимо** друг от друга. Мы можем:

1.  Спроектировать регулятор, как будто `x` нам известен, и найти `K`.
2.  Спроектировать наблюдатель, чтобы получить хорошую оценку `✰`, и найти `L`.
3.  Затем просто соединить их, используя в законе управления оценку состояния вместо реального: `u = -K✰`.

При этом собственные числа объединенной системы (регулятор + наблюдатель) будут просто объединением полюсов регулятора (собственных чисел `A-BK`) и полюсов наблюдателя (собственных чисел `A-LC`). Они не влияют друг на друга.

**Практическая рекомендация:** Динамика наблюдателя должна быть в 2-5 раз быстрее динамики самого объекта управления. Это значит, что полюсы наблюдателя должны лежать значительно левее полюсов регулятора на комплексной плоскости. Это гарантирует, что оценка `✰` сойдется к `x` быстрее, чем система успеет среагировать на управляющее воздействие, и регулятор будет работать почти так же хорошо, как если бы он использовал истинное состояние `x`.

## 6. Ссылки на материалы

1.  State observer // Wikipedia. – [URL: https://en.wikipedia.org/wiki/State_observer](https://en.wikipedia.org/wiki/State_observer)
2.  Наблюдатели состояния. Наблюдатель Люенбергера // T-Flex. – [URL: https://www.tflex.ru/study/au/au_10.php](https://www.tflex.ru/study/au/au_10.php)
3.  Control Systems | Observers and Controllers // The University of Manchester. – [URL: https://www.youtube.com/watch?v=FxU-yprh94E](https://www.youtube.com/watch?v=FxU-yprh94E)


## 7. Практический пример на Python

Реализуем наблюдатель Люенбергера и продемонстрируем принцип разделения на практике.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import eigvals, solve
import sympy as sp

def luenberger_observer_design(A, C, desired_observer_poles):
    """
    Синтез наблюдателя Люенбергера методом размещения полюсов
    """
    n = A.shape[0]
    
    # Проверка наблюдаемости
    O = ctrl.obsv(A, C)
    if np.linalg.matrix_rank(O) != n:
        raise ValueError("Система не полностью наблюдаема!")
    
    # Используем дуальность: размещение полюсов для (A^T, C^T)
    # эквивалентно синтезу наблюдателя для (A, C)
    L_T = ctrl.place(A.T, C.T, desired_observer_poles)
    L = L_T.T
    
    return L

def simulate_observer_controller_system():
    """
    Полное моделирование системы с регулятором и наблюдателем
    """
    
    print("СИСТЕМА С РЕГУЛЯТОРОМ И НАБЛЮДАТЕЛЕМ")
    print("="*60)
    
    # Исходная система (модель самолета)
    A = np.array([[0, 1],
                  [-4, -1]])
    
    B = np.array([[0],
                  [1]])
    
    C = np.array([[1, 0]])  # измеряем только положение (не скорость)
    
    D = np.array([[0]])
    
    print("Исходная система:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"C = \n{C}")
    
    # Проверка управляемости и наблюдаемости
    P = ctrl.ctrb(A, B)
    O = ctrl.obsv(A, C)
    
    controllable = np.linalg.matrix_rank(P) == A.shape[0]
    observable = np.linalg.matrix_rank(O) == A.shape[0]
    
    print(f"\nУправляемость: {'✓' if controllable else '✗'}")
    print(f"Наблюдаемость: {'✓' if observable else '✗'}")
    
    if not (controllable and observable):
        print("Система не подходит для синтеза!")
        return
    
    # Синтез регулятора
    controller_poles = [-2, -3]  # желаемые полюсы замкнутой системы
    K = ctrl.place(A, B, controller_poles)
    
    print(f"\nРегулятор:")
    print(f"Желаемые полюсы регулятора: {controller_poles}")
    print(f"Матрица K: {K}")
    
    # Синтез наблюдателя (полюсы в 3 раза быстрее)
    observer_poles = [-6, -9]  # быстрее полюсов регулятора
    L = luenberger_observer_design(A, C, observer_poles)
    
    print(f"\nНаблюдатель:")
    print(f"Желаемые полюсы наблюдателя: {observer_poles}")
    print(f"Матрица L: {L}")
    
    # Проверка фактических полюсов
    A_cl = A - B @ K  # полюсы регулятора
    A_obs = A - L @ C  # полюсы наблюдателя
    
    actual_controller_poles = eigvals(A_cl)
    actual_observer_poles = eigvals(A_obs)
    
    print(f"\nФактические полюсы регулятора: {actual_controller_poles}")
    print(f"Фактические полюсы наблюдателя: {actual_observer_poles}")
    
    return A, B, C, D, K, L

def simulate_complete_system(A, B, C, D, K, L):
    """
    Моделирование полной системы: объект + регулятор + наблюдатель
    """
    
    # Расширенная система: [x; x_hat]
    # где x - истинное состояние, x_hat - оценка состояния
    
    n = A.shape[0]  # размерность состояния
    
    # Матрица расширенной системы
    A_ext = np.block([
        [A - B @ K,     B @ K],      # уравнение для x
        [np.zeros((n, n)), A - L @ C]  # уравнение для x_hat
    ])
    
    # Входная матрица (возмущения)
    B_ext = np.block([
        [B],
        [np.zeros((n, 1))]
    ])
    
    # Выходная матрица
    C_ext = np.block([
        [C, np.zeros((1, n))],  # истинный выход
        [np.zeros((1, n)), C]   # оценка выхода
    ])
    
    print(f"\n{'='*60}")
    print("МОДЕЛИРОВАНИЕ ПОЛНОЙ СИСТЕМЫ")
    print(f"{'='*60}")
    
    # Анализ полюсов расширенной системы
    poles_extended = eigvals(A_ext)
    print(f"Полюсы расширенной системы: {poles_extended}")
    
    # Проверка принципа разделения
    controller_poles = eigvals(A - B @ K)
    observer_poles = eigvals(A - L @ C)
    combined_poles = np.concatenate([controller_poles, observer_poles])
    
    print(f"Полюсы регулятора: {controller_poles}")
    print(f"Полюсы наблюдателя: {observer_poles}")
    print(f"Объединенные полюсы: {combined_poles}")
    
    # Проверка принципа разделения
    separation_check = np.allclose(np.sort(poles_extended), np.sort(combined_poles))
    print(f"Принцип разделения выполняется: {'✓' if separation_check else '✗'}")
    
    return A_ext, B_ext, C_ext

def compare_control_scenarios(A, B, C, D, K, L):
    """
    Сравнение различных сценариев управления
    """
    
    plt.figure(figsize=(16, 12))
    
    # Время моделирования
    t = np.linspace(0, 8, 1000)
    
    # Начальные условия
    x0_true = np.array([1, 0])  # истинное начальное состояние
    x0_est = np.array([0, 0])   # начальная оценка (неточная)
    
    # Сценарий 1: Идеальная обратная связь (знаем истинное состояние)
    print("Моделирование сценария 1: Идеальная обратная связь...")
    A_ideal = A - B @ K
    sys_ideal = ctrl.ss(A_ideal, np.zeros((2, 1)), C, np.zeros((1, 1)))
    
    t1, y1, x1 = ctrl.initial_response(sys_ideal, t, X0=x0_true, return_x=True)
    u1 = -K @ x1  # управляющий сигнал
    
    # Сценарий 2: Только наблюдатель (без управления)
    print("Моделирование сценария 2: Только наблюдатель...")
    
    # Расширенная система для наблюдателя
    A_obs_ext = np.block([
        [A, np.zeros((2, 2))],
        [L @ C, A - L @ C]
    ])
    
    C_obs_ext = np.block([
        [C, np.zeros((1, 2))],  # истинный выход
        [np.zeros((1, 2)), C]   # оценка выхода
    ])
    
    sys_obs = ctrl.ss(A_obs_ext, np.zeros((4, 1)), C_obs_ext, np.zeros((2, 1)))
    x0_obs_ext = np.concatenate([x0_true, x0_est])
    
    t2, y2, x2 = ctrl.initial_response(sys_obs, t, X0=x0_obs_ext, return_x=True)
    
    # Сценарий 3: Полная система (регулятор + наблюдатель)
    print("Моделирование сценария 3: Полная система...")
    
    # Моделирование полной системы численно
    def full_system_dynamics(t, state):
        x = state[:2]      # истинное состояние
        x_hat = state[2:]  # оценка состояния
        
        y = C @ x          # измерение
        u = -K @ x_hat     # управление на основе оценки
        
        dx_dt = A @ x + B @ u
        dx_hat_dt = A @ x_hat + B @ u + L @ (y - C @ x_hat)
        
        return np.concatenate([dx_dt, dx_hat_dt])
    
    # Численное интегрирование
    from scipy.integrate import solve_ivp
    
    x0_full = np.concatenate([x0_true, x0_est])
    sol = solve_ivp(full_system_dynamics, [0, t[-1]], x0_full, t_eval=t, rtol=1e-8)
    
    x3 = sol.y[:2, :]      # истинное состояние
    x_hat3 = sol.y[2:, :]  # оценка состояния
    y3 = C @ x3            # выход
    u3 = -K @ x_hat3       # управление
    
    # Построение графиков
    
    # 1. Состояния системы
    plt.subplot(3, 3, 1)
    plt.plot(t1, x1[0, :], 'b-', label='Идеальная ОС', linewidth=2)
    plt.plot(t, x3[0, :], 'r--', label='С наблюдателем', linewidth=2)
    plt.plot(t2, x2[0, :], 'g:', label='Без управления', linewidth=2)
    plt.title('Положение x₁')
    plt.xlabel('Время (с)')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.plot(t1, x1[1, :], 'b-', label='Идеальная ОС', linewidth=2)
    plt.plot(t, x3[1, :], 'r--', label='С наблюдателем', linewidth=2)
    plt.plot(t2, x2[1, :], 'g:', label='Без управления', linewidth=2)
    plt.title('Скорость x₂')
    plt.xlabel('Время (с)')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True)
    
    # 2. Управляющие сигналы
    plt.subplot(3, 3, 3)
    plt.plot(t1, u1[0, :], 'b-', label='Идеальная ОС', linewidth=2)
    plt.plot(t, u3[0, :], 'r--', label='С наблюдателем', linewidth=2)
    plt.title('Управляющий сигнал')
    plt.xlabel('Время (с)')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    
    # 3. Ошибка наблюдения
    plt.subplot(3, 3, 4)
    error1 = x3[0, :] - x_hat3[0, :]
    error2 = x3[1, :] - x_hat3[1, :]
    plt.plot(t, error1, 'r-', label='Ошибка x₁')
    plt.plot(t, error2, 'b-', label='Ошибка x₂')
    plt.title('Ошибка наблюдения')
    plt.xlabel('Время (с)')
    plt.ylabel('x - x̂')
    plt.legend()
    plt.grid(True)
    
    # 4. Истинное состояние vs оценка
    plt.subplot(3, 3, 5)
    plt.plot(t, x3[0, :], 'b-', label='Истинное x₁', linewidth=2)
    plt.plot(t, x_hat3[0, :], 'r--', label='Оценка x̂₁', linewidth=2)
    plt.title('Сравнение: истинное vs оценка (x₁)')
    plt.xlabel('Время (с)')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 6)
    plt.plot(t, x3[1, :], 'b-', label='Истинное x₂', linewidth=2)
    plt.plot(t, x_hat3[1, :], 'r--', label='Оценка x̂₂', linewidth=2)
    plt.title('Сравнение: истинное vs оценка (x₂)')
    plt.xlabel('Время (с)')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True)
    
    # 5. Фазовые портреты
    plt.subplot(3, 3, 7)
    plt.plot(x1[0, :], x1[1, :], 'b-', label='Идеальная ОС', linewidth=2)
    plt.plot(x3[0, :], x3[1, :], 'r--', label='С наблюдателем', linewidth=2)
    plt.plot(x0_true[0], x0_true[1], 'ko', markersize=8, label='Начальная точка')
    plt.title('Фазовые портреты')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True)
    
    # 6. Карта полюсов
    plt.subplot(3, 3, 8)
    
    # Полюсы различных систем
    poles_controller = eigvals(A - B @ K)
    poles_observer = eigvals(A - L @ C)
    poles_original = eigvals(A)
    
    plt.plot(np.real(poles_original), np.imag(poles_original), 'ko', 
             markersize=10, label='Исходная система')
    plt.plot(np.real(poles_controller), np.imag(poles_controller), 'bs', 
             markersize=8, label='Регулятор')
    plt.plot(np.real(poles_observer), np.imag(poles_observer), 'r^', 
             markersize=8, label='Наблюдатель')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvspan(-12, 0, alpha=0.2, color='green')
    plt.title('Карта полюсов')
    plt.xlabel('Действительная часть')
    plt.ylabel('Мнимая часть')
    plt.legend()
    plt.grid(True)
    
    # 7. Анализ качества
    plt.subplot(3, 3, 9)
    
    # Время регулирования
    final_value_ideal = x1[0, -1]
    final_value_observer = x3[0, -1]
    
    # 2% критерий
    settling_idx_ideal = np.where(np.abs(x1[0, :] - final_value_ideal) <= 0.02 * abs(final_value_ideal))[0]
    settling_idx_observer = np.where(np.abs(x3[0, :] - final_value_observer) <= 0.02 * abs(final_value_observer))[0]
    
    settling_time_ideal = t1[settling_idx_ideal[0]] if len(settling_idx_ideal) > 0 else t1[-1]
    settling_time_observer = t[settling_idx_observer[0]] if len(settling_idx_observer) > 0 else t[-1]
    
    systems = ['Идеальная ОС', 'С наблюдателем']
    settling_times = [settling_time_ideal, settling_time_observer]
    
    plt.bar(systems, settling_times, alpha=0.7)
    plt.title('Время регулирования')
    plt.ylabel('Время (с)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод результатов
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print(f"{'='*60}")
    print(f"Время регулирования (идеальная ОС): {settling_time_ideal:.2f} с")
    print(f"Время регулирования (с наблюдателем): {settling_time_observer:.2f} с")
    print(f"Разница: {abs(settling_time_observer - settling_time_ideal):.2f} с")
    
    # Максимальная ошибка наблюдения
    max_error = np.max(np.abs(error1))
    print(f"Максимальная ошибка наблюдения: {max_error:.4f}")

def observer_pole_sensitivity():
    """Анализ влияния размещения полюсов наблюдателя"""
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ВЛИЯНИЯ ПОЛЮСОВ НАБЛЮДАТЕЛЯ")
    print(f"{'='*60}")
    
    A = np.array([[0, 1], [-4, -1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    # Фиксированный регулятор
    K = ctrl.place(A, B, [-2, -3])
    
    # Различные варианты полюсов наблюдателя
    observer_configs = {
        'Медленный': [-1, -1.5],
        'Умеренный': [-4, -6],
        'Быстрый': [-8, -12],
        'Очень быстрый': [-15, -20]
    }
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (name, poles) in enumerate(observer_configs.items()):
        L = luenberger_observer_design(A, C, poles)
        
        # Моделирование
        t = np.linspace(0, 5, 500)
        x0_true = np.array([1, 0])
        x0_est = np.array([0, 0])
        
        # Численное моделирование
        def system_dynamics(t, state):
            x = state[:2]
            x_hat = state[2:]
            
            y = C @ x
            u = -K @ x_hat
            
            dx_dt = A @ x + B @ u
            dx_hat_dt = A @ x_hat + B @ u + L @ (y - C @ x_hat)
            
            return np.concatenate([dx_dt, dx_hat_dt])
        
        from scipy.integrate import solve_ivp
        x0_full = np.concatenate([x0_true, x0_est])
        sol = solve_ivp(system_dynamics, [0, t[-1]], x0_full, t_eval=t, rtol=1e-8)
        
        x_true = sol.y[:2, :]
        x_hat = sol.y[2:, :]
        error = x_true - x_hat
        
        # График ошибки наблюдения
        plt.subplot(2, 2, 1)
        plt.plot(t, error[0, :], color=colors[i], label=f'{name} ({poles})')
        
        plt.subplot(2, 2, 2)
        plt.plot(t, error[1, :], color=colors[i], label=f'{name}')
        
        # График состояний
        plt.subplot(2, 2, 3)
        plt.plot(t, x_true[0, :], color=colors[i], label=f'{name}')
        
        # Управляющий сигнал
        plt.subplot(2, 2, 4)
        u = -K @ x_hat
        plt.plot(t, u[0, :], color=colors[i], label=f'{name}')
    
    plt.subplot(2, 2, 1)
    plt.title('Ошибка наблюдения x₁')
    plt.xlabel('Время (с)')
    plt.ylabel('e₁ = x₁ - x̂₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.title('Ошибка наблюдения x₂')
    plt.xlabel('Время (с)')
    plt.ylabel('e₂ = x₂ - x̂₂')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.title('Состояние x₁')
    plt.xlabel('Время (с)')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.title('Управляющий сигнал')
    plt.xlabel('Время (с)')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Синтез системы
    A, B, C, D, K, L = simulate_observer_controller_system()
    
    # Анализ расширенной системы
    A_ext, B_ext, C_ext = simulate_complete_system(A, B, C, D, K, L)
    
    # Сравнение сценариев
    compare_control_scenarios(A, B, C, D, K, L)
    
    # Анализ чувствительности
    observer_pole_sensitivity()
    
    print(f"\n{'='*60}")
    print("ОСНОВНЫЕ ВЫВОДЫ")
    print(f"{'='*60}")
    print("1. Принцип разделения позволяет независимо проектировать регулятор и наблюдатель")
    print("2. Наблюдатель должен быть быстрее основной системы (в 3-5 раз)")
    print("3. Слишком быстрые полюсы наблюдателя могут усиливать шум")
    print("4. Система с наблюдателем близка по качеству к идеальной обратной связи")
    print("5. Ошибка наблюдения экспоненциально затухает")
```

Этот пример демонстрирует:

1. **Синтез наблюдателя Люенбергера** с использованием принципа дуальности
2. **Принцип разделения** - независимое проектирование регулятора и наблюдателя
3. **Сравнение различных сценариев управления** (идеальная обратная связь vs наблюдатель)
4. **Анализ ошибки наблюдения** и ее динамики
5. **Влияние размещения полюсов наблюдателя** на качество системы
6. **Численное моделирование** полной нелинейной системы с обратной связью
