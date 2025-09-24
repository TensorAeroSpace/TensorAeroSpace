# Урок 5 — Синтез в пространстве состояний: метод размещения полюсов

## 1. От анализа к синтезу

На предыдущих занятиях мы научились анализировать системы: представлять их в пространстве состояний, проверять на управляемость и наблюдаемость, а также определять их устойчивость по расположению полюсов (собственных чисел матрицы A). 

**Синтез** – это обратная задача. Мы не просто анализируем то, что есть, а целенаправленно создаем систему управления (регулятор), которая обеспечит **желаемое** поведение замкнутой системы. В частности, мы хотим, чтобы замкнутая система была устойчивой и имела требуемые характеристики переходного процесса (быстродействие, колебательность и т.д.).

## 2. Управление с обратной связью по состоянию

Основная идея синтеза в пространстве состояний – это использование **обратной связи по вектору состояния**. Мы измеряем (или оцениваем) все переменные состояния `x(t)` и на их основе формируем управляющее воздействие `u(t)`.

Закон управления имеет вид:

```
u = -Kx
```

где `K` – это **матрица коэффициентов обратной связи** (или матрица усиления регулятора). Наша задача – найти такую матрицу `K`, которая обеспечит желаемые свойства системы.

Подставим этот закон управления в уравнение состояния исходной системы:

```
ẋ = Ax + Bu
ẋ = Ax + B(-Kx) = Ax - BKx
ẋ = (A - BK)x
```

Мы получили уравнение **замкнутой системы**. Динамика этой системы теперь определяется новой матрицей `A_cl = (A - BK)`. Устойчивость и характер переходных процессов в замкнутой системе зависят от собственных чисел матрицы `A_cl`.

## 3. Метод размещения полюсов (Pole Placement)

**Метод размещения полюсов** заключается в том, чтобы выбрать матрицу `K` таким образом, чтобы собственные числа матрицы `A_cl = (A - BK)` оказались в заранее заданных, **желаемых** точках комплексной плоскости. Эти желаемые собственные числа и есть **желаемые полюсы** замкнутой системы.

**Теорема:** Если система `(A, B)` полностью управляема, то мы можем разместить полюсы замкнутой системы `(A - BK)` в **любых** желаемых местах комплексной плоскости, выбрав соответствующую матрицу `K`.

Это фундаментальный результат. Управляемость – это необходимое и достаточное условие для того, чтобы мы могли полностью контролировать динамику линейной системы с помощью обратной связи по состоянию.

### 3.1. Как выбрать желаемые полюсы?

Выбор желаемых полюсов определяет динамику замкнутой системы:

*   **Расположение на действительной оси:** Чем левее на действительной оси находятся полюсы, тем быстрее затухают переходные процессы (система более "быстрая").
*   **Комплексно-сопряженные полюсы:** Пара комплексно-сопряженных полюсов `λ = -σ ± jω` порождает колебания. 
    *   `σ` (действительная часть) определяет **затухание** колебаний. Чем больше `σ`, тем быстрее затухают колебания.
    *   `ω` (мнимая часть) определяет **частоту** колебаний.

Часто желаемые полюсы выбирают на основе требований к переходному процессу, например, используя стандартные формы (прототипы) вроде фильтра Баттерворта, которые обеспечивают хороший компромисс между быстродействием и колебательностью.

### 3.2. Формула Аккермана

Для систем с одним входом (когда `u` – скаляр) существует прямая формула для вычисления матрицы `K`, известная как **формула Аккермана**.

1.  Задаем желаемые полюсы `p₁, p₂, ..., pₙ`.
2.  Составляем желаемый характеристический полином: `α(s) = (s - p₁)(s - p₂)...(s - pₙ)`.
3.  Находим матрицу управляемости `P = [B, AB, ..., Aⁿ⁻¹B]`.
4.  Вычисляем матрицу `K` по формуле:

    ```
    K = [0 0 ... 1] * P⁻¹ * α(A)
    ```

    где `α(A)` – это матричный полином, получаемый подстановкой матрицы `A` в желаемый характеристический полином `α(s)`.

## 4. Ограничения

Метод размещения полюсов – мощный инструмент, но он требует знания **всех** переменных состояния `x(t)`. На практике это не всегда возможно: некоторые переменные могут быть физически недоступны для измерения. Эта проблема решается с помощью **наблюдателей состояния**, которые мы рассмотрим на следующем семинаре.

## 5. Ссылки на материалы

1.  Pole placement // Wikipedia. – [URL: https://en.wikipedia.org/wiki/Pole_placement](https://en.wikipedia.org/wiki/Pole_placement)
2.  State-Space Methods for Controller Design // University of Michigan. – [URL: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=ControlStateSpace](https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=ControlStateSpace)
3.  Синтез модального регулятора (метод размещения полюсов) // Habr. – [URL: https://habr.com/ru/post/562152/](https://habr.com/ru/post/562152/)


## 6. Практический пример на Python

Реализуем метод размещения полюсов для системы управления самолетом и сравним различные варианты размещения.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import solve, eigvals
import sympy as sp

def ackermann_formula(A, B, desired_poles):
    """
    Формула Аккермана для вычисления матрицы обратной связи K
    для систем с одним входом (SISO)
    """
    n = A.shape[0]
    
    # Проверка управляемости
    P = ctrl.ctrb(A, B)
    if np.linalg.matrix_rank(P) != n:
        raise ValueError("Система не полностью управляема!")
    
    # Построение желаемого характеристического полинома
    # α(s) = (s - p1)(s - p2)...(s - pn)
    poly_coeffs = np.poly(desired_poles)  # коэффициенты полинома
    
    # Вычисление α(A) - матричного полинома
    alpha_A = np.zeros_like(A)
    for i, coeff in enumerate(poly_coeffs):
        alpha_A += coeff * np.linalg.matrix_power(A, len(poly_coeffs) - 1 - i)
    
    # Формула Аккермана: K = [0 0 ... 1] * P^(-1) * α(A)
    e_n = np.zeros((1, n))
    e_n[0, -1] = 1  # вектор [0 0 ... 1]
    
    K = e_n @ np.linalg.inv(P) @ alpha_A
    
    return K

def place_poles_comparison():
    """Сравнение различных вариантов размещения полюсов"""
    
    # Исходная система (упрощенная модель самолета)
    A = np.array([[0, 1],
                  [-2, -0.5]])  # слабо задемпфированная система
    
    B = np.array([[0],
                  [1]])
    
    C = np.array([[1, 0]])  # измеряем положение
    D = np.array([[0]])
    
    print("ИСХОДНАЯ СИСТЕМА")
    print("="*50)
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    
    # Анализ исходной системы
    original_poles = eigvals(A)
    print(f"\nИсходные полюсы: {original_poles}")
    
    # Проверка управляемости
    P = ctrl.ctrb(A, B)
    controllable = np.linalg.matrix_rank(P) == A.shape[0]
    print(f"Управляемость: {'✓' if controllable else '✗'}")
    
    if not controllable:
        print("Система не управляема! Размещение полюсов невозможно.")
        return
    
    # Варианты размещения полюсов
    pole_sets = {
        'Быстрое затухание': [-5, -6],
        'Критическое демпфирование': [-3, -3],
        'Колебательная система': [-1+2j, -1-2j],
        'Очень быстрая': [-10, -12]
    }
    
    results = {}
    
    print(f"\n{'='*60}")
    print("СИНТЕЗ РЕГУЛЯТОРОВ")
    print(f"{'='*60}")
    
    for name, poles in pole_sets.items():
        print(f"\n{name}:")
        print(f"Желаемые полюсы: {poles}")
        
        # Вычисление матрицы K
        try:
            if len(set(poles)) == 1:  # повторяющиеся полюсы
                K = ctrl.place(A, B, poles)
            else:
                K = ackermann_formula(A, B, poles)
            
            # Проверка результата
            A_cl = A - B @ K
            actual_poles = eigvals(A_cl)
            
            print(f"Матрица K: {K}")
            print(f"Фактические полюсы: {actual_poles}")
            
            results[name] = {
                'K': K,
                'poles': actual_poles,
                'A_cl': A_cl
            }
            
        except Exception as e:
            print(f"Ошибка: {e}")
            results[name] = None
    
    return A, B, C, D, results

def simulate_closed_loop_systems(A, B, C, D, results):
    """Моделирование замкнутых систем с различными регуляторами"""
    
    # Исходная система
    sys_open = ctrl.ss(A, B, C, D)
    
    plt.figure(figsize=(15, 12))
    
    # Время моделирования
    t = np.linspace(0, 5, 1000)
    
    # 1. Переходные процессы
    plt.subplot(3, 2, 1)
    
    # Исходная система
    t_orig, y_orig = ctrl.step_response(sys_open, t)
    plt.plot(t_orig, y_orig, 'k--', linewidth=2, label='Исходная система')
    
    # Замкнутые системы
    colors = ['red', 'blue', 'green', 'orange']
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            # Замкнутая система
            A_cl = result['A_cl']
            sys_cl = ctrl.ss(A_cl, B, C, D)
            
            try:
                t_cl, y_cl = ctrl.step_response(sys_cl, t)
                plt.plot(t_cl, y_cl, color=colors[i], label=name)
            except:
                print(f"Ошибка моделирования для {name}")
    
    plt.title('Переходные процессы')
    plt.xlabel('Время (с)')
    plt.ylabel('Выход')
    plt.legend()
    plt.grid(True)
    
    # 2. Карта полюсов
    plt.subplot(3, 2, 2)
    
    # Исходные полюсы
    orig_poles = eigvals(A)
    plt.plot(np.real(orig_poles), np.imag(orig_poles), 'ks', 
             markersize=10, label='Исходные полюсы')
    
    # Полюсы замкнутых систем
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            poles = result['poles']
            plt.plot(np.real(poles), np.imag(poles), 'o', 
                    color=colors[i], markersize=8, label=name)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvspan(-15, 0, alpha=0.2, color='green', label='Область устойчивости')
    plt.title('Карта полюсов')
    plt.xlabel('Действительная часть')
    plt.ylabel('Мнимая часть')
    plt.legend()
    plt.grid(True)
    
    # 3. Управляющие сигналы
    plt.subplot(3, 2, 3)
    
    # Начальные условия для демонстрации
    x0 = np.array([1, 0])  # начальное отклонение
    
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            A_cl = result['A_cl']
            K = result['K']
            
            # Моделирование с начальными условиями
            sys_cl = ctrl.ss(A_cl, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
            
            try:
                t_sim, x_sim = ctrl.initial_response(sys_cl, t, X0=x0)
                u_sim = -K @ x_sim  # управляющий сигнал
                plt.plot(t_sim, u_sim[0], color=colors[i], label=name)
            except:
                print(f"Ошибка моделирования управления для {name}")
    
    plt.title('Управляющие сигналы')
    plt.xlabel('Время (с)')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid(True)
    
    # 4. Фазовые портреты
    plt.subplot(3, 2, 4)
    
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            A_cl = result['A_cl']
            
            # Моделирование с начальными условиями
            sys_cl = ctrl.ss(A_cl, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
            
            try:
                t_sim, x_sim = ctrl.initial_response(sys_cl, t, X0=x0)
                plt.plot(x_sim[0], x_sim[1], color=colors[i], label=name)
            except:
                print(f"Ошибка построения фазового портрета для {name}")
    
    plt.title('Фазовые портреты')
    plt.xlabel('x1 (положение)')
    plt.ylabel('x2 (скорость)')
    plt.legend()
    plt.grid(True)
    
    # 5. Анализ качества (время регулирования, перерегулирование)
    plt.subplot(3, 2, 5)
    
    settling_times = []
    overshoots = []
    names = []
    
    for name, result in results.items():
        if result is not None:
            A_cl = result['A_cl']
            sys_cl = ctrl.ss(A_cl, B, C, D)
            
            try:
                t_step, y_step = ctrl.step_response(sys_cl, t)
                
                # Время регулирования (2% критерий)
                final_value = y_step[-1]
                settling_idx = np.where(np.abs(y_step - final_value) <= 0.02 * abs(final_value))[0]
                settling_time = t_step[settling_idx[0]] if len(settling_idx) > 0 else t[-1]
                
                # Перерегулирование
                max_value = np.max(y_step)
                overshoot = (max_value - final_value) / final_value * 100 if final_value != 0 else 0
                
                settling_times.append(settling_time)
                overshoots.append(max(0, overshoot))
                names.append(name)
                
            except:
                pass
    
    x_pos = np.arange(len(names))
    plt.bar(x_pos, settling_times, alpha=0.7)
    plt.title('Время регулирования')
    plt.xlabel('Тип регулятора')
    plt.ylabel('Время (с)')
    plt.xticks(x_pos, names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Перерегулирование
    plt.subplot(3, 2, 6)
    plt.bar(x_pos, overshoots, alpha=0.7, color='orange')
    plt.title('Перерегулирование')
    plt.xlabel('Тип регулятора')
    plt.ylabel('Перерегулирование (%)')
    plt.xticks(x_pos, names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def robust_pole_placement():
    """Демонстрация влияния неточностей модели на размещение полюсов"""
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ РОБАСТНОСТИ")
    print(f"{'='*60}")
    
    # Номинальная система
    A_nom = np.array([[0, 1],
                      [-2, -0.5]])
    B_nom = np.array([[0], [1]])
    
    # Желаемые полюсы
    desired_poles = [-2, -3]
    
    # Синтез регулятора для номинальной системы
    K = ackermann_formula(A_nom, B_nom, desired_poles)
    print(f"Регулятор для номинальной системы: K = {K}")
    
    # Анализ чувствительности к параметрам
    uncertainties = np.linspace(-0.5, 0.5, 21)  # ±50% неопределенность
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for delta in uncertainties[::4]:  # каждый 4-й для наглядности
        A_pert = A_nom + delta * np.array([[0, 0], [0.5, 0.1]])  # возмущение параметров
        A_cl = A_pert - B_nom @ K
        poles = eigvals(A_cl)
        
        plt.plot(np.real(poles), np.imag(poles), 'o', alpha=0.6, 
                label=f'δ = {delta:.1f}' if abs(delta) < 0.3 else '')
    
    # Номинальные полюсы
    A_cl_nom = A_nom - B_nom @ K
    poles_nom = eigvals(A_cl_nom)
    plt.plot(np.real(poles_nom), np.imag(poles_nom), 'rs', markersize=10, 
             label='Номинальные')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Чувствительность полюсов к неопределенности')
    plt.xlabel('Действительная часть')
    plt.ylabel('Мнимая часть')
    plt.legend()
    plt.grid(True)
    
    # Анализ устойчивости при неопределенности
    plt.subplot(2, 2, 2)
    stable_count = []
    
    for delta in uncertainties:
        A_pert = A_nom + delta * np.array([[0, 0], [0.5, 0.1]])
        A_cl = A_pert - B_nom @ K
        poles = eigvals(A_cl)
        
        # Проверка устойчивости
        is_stable = all(np.real(pole) < 0 for pole in poles)
        stable_count.append(1 if is_stable else 0)
    
    plt.plot(uncertainties * 100, stable_count, 'bo-')
    plt.title('Робастная устойчивость')
    plt.xlabel('Неопределенность параметров (%)')
    plt.ylabel('Устойчивость (1=да, 0=нет)')
    plt.grid(True)
    plt.ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Синтез регуляторов
    A, B, C, D, results = place_poles_comparison()
    
    # Моделирование и сравнение
    simulate_closed_loop_systems(A, B, C, D, results)
    
    # Анализ робастности
    robust_pole_placement()
    
    print(f"\n{'='*60}")
    print("ВЫВОДЫ")
    print(f"{'='*60}")
    print("1. Размещение полюсов позволяет точно задать динамику системы")
    print("2. Быстрые полюсы дают быстрый отклик, но требуют больших управляющих сигналов")
    print("3. Колебательные полюсы могут давать перерегулирование")
    print("4. Необходимо учитывать робастность при неопределенности параметров")
    print("5. Компромисс между быстродействием и энергозатратами")
```

Этот пример демонстрирует:

1. **Реализацию формулы Аккермана** для вычисления матрицы обратной связи
2. **Сравнение различных стратегий размещения полюсов** и их влияние на динамику системы
3. **Анализ качества переходных процессов** (время регулирования, перерегулирование)
4. **Визуализацию результатов** через переходные процессы, карты полюсов и фазовые портреты
5. **Анализ робастности** - влияние неточностей модели на работу регулятора
6. **Практические рекомендации** по выбору желаемых полюсов
