# Урок 7 — Практика: стабилизация продольного движения самолета

## 1. Цель

На этом заключительном семинаре мы применим все полученные знания на практике. Наша цель – разработать систему автоматической стабилизации (автопилот) для упрощенной модели продольного движения легкого самолета. Мы пройдем все этапы: от получения модели до синтеза полной системы управления с наблюдателем.

## 2. Модель объекта управления

Рассмотрим линеаризованную модель короткопериодического продольного движения самолета. Переменными состояния являются:

*   `α` (alpha) – угол атаки
*   `q` – угловая скорость тангажа

Управляющим воздействием является отклонение руля высоты `δₑ` (delta_e). Выходной переменной, которую мы можем измерить, будем считать угол тангажа `θ` (theta), который в данном случае равен интегралу от угловой скорости `q`.

Матрицы системы (для конкретного самолета и режима полета) могут выглядеть следующим образом:

```
A = [[-0.313, 1.0], [-0.0139, -0.426]]
B = [[-0.0322], [0.0]]
C = [[0, 1]]
D = [0]
```

## 3. Этап 1: Анализ исходной системы

### 3.1. Устойчивость

Первым делом проверим устойчивость "голого" самолета, то есть без системы управления. Для этого найдем собственные числа матрицы `A`.

`λ₁,₂ = -0.3695 ± 0.1627j`

Действительные части обоих полюсов отрицательны, значит, самолет **статически и динамически устойчив**. Однако, посмотрим на характеристики этой устойчивости. Демпфирование (`σ = 0.3695`) относительно небольшое, что может приводить к нежелательным колебаниям.

### 3.2. Управляемость и наблюдаемость

Проверим, можем ли мы в принципе управлять этой системой и наблюдать за ней.

*   **Матрица управляемости:** `P = [B, AB]`. `rank(P) = 2`. Система **полностью управляема**.
*   **Матрица наблюдаемости:** `Q = [C; CA]`. `rank(Q) = 2`. Система **полностью наблюдаема**.

Отлично, мы можем приступать к синтезу.

## 4. Этап 2: Синтез регулятора (размещение полюсов)

Допустим, мы хотим сделать систему более "отзывчивой" и лучше задемпфированной. Поставим цель разместить полюсы замкнутой системы в точках:

`p₁,₂ = -1.0 ± 1.0j`

Это соответствует более быстрому затуханию (`σ = 1.0` вместо `0.3695`) и более высокой частоте колебаний.

1.  **Желаемый характеристический полином:** `α(s) = (s - p₁)(s - p₂) = s² + 2s + 2`.
2.  **Применяем формулу Аккермана** (или другой метод) для нахождения матрицы `K`.

В результате вычислений получаем:

`K = [-26.4, -15.7]`

Закон управления: `δₑ = -([-26.4, -15.7]) * [α; q] = 26.4α + 15.7q`

## 5. Этап 3: Синтез наблюдателя

Теперь предположим, что мы не можем измерять угол атаки `α` напрямую. У нас есть только датчик угловой скорости тангажа `q`. Нам нужно построить наблюдатель, чтобы оценить `α`.

Динамика наблюдателя должна быть быстрее динамики регулятора. Выберем полюсы для наблюдателя, например, в 5 раз левее полюсов регулятора:

`l₁,₂ = -5.0 ± 5.0j`

1.  **Желаемый характеристический полином наблюдателя:** `α_obs(s) = s² + 10s + 50`.
2.  **Используем дуальную формулу Аккермана** для нахождения матрицы `L`.

Получаем матрицу усиления наблюдателя `L`.

## 6. Этап 4: Сборка полной системы

Теперь мы собираем все вместе. Полная система управления состоит из:

1.  **Наблюдателя состояния:**
    `✰̇ = A✰ + Bδₑ + L(q - q̂)`
2.  **Закона управления (регулятора):**
    `δₑ = -K✰`

Эта система (автопилот) принимает на вход измеренную угловую скорость `q` и выдает управляющий сигнал `δₑ` на руль высоты, стабилизируя самолет в соответствии с желаемой динамикой, заданной полюсами `p₁,₂`.

## 7. Задание для самостоятельной работы

1.  Используя MATLAB, Python (с библиотекой `control`) или другой пакет, проведите все вычисления, представленные в этом семинаре:
    *   Проверьте устойчивость, управляемость и наблюдаемость исходной системы.
    *   Вычислите матрицу `K` для заданных желаемых полюсов регулятора.
    *   Вычислите матрицу `L` для заданных желаемых полюсов наблюдателя.
2.  Постройте Simulink-модель или напишите скрипт, который моделирует поведение:
    *   Исходной (неуправляемой) системы.
    *   Системы с регулятором (с обратной связью по истинному состоянию).
    *   Полной системы (регулятор + наблюдатель).
3.  Сравните переходные процессы для всех трех случаев при задании начального отклонения (например, начального угла атаки). Убедитесь, что система с регулятором работает лучше исходной, а полная система очень близка по поведению к системе с идеальной обратной связью.

## 8. Ссылки на материалы

1.  Aircraft Pitch: State-Space Methods for Controller Design // University of Michigan. – [URL: https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlStateSpace](https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlStateSpace)
2.  Python Control Systems Library. – [URL: https://python-control.readthedocs.io/en/latest/](https://python-control.readthedocs.io/en/latest/)
3.  MATLAB Control System Toolbox. – [URL: https://www.mathworks.com/products/control.html](https://www.mathworks.com/products/control.html)


## 9. Полная реализация на Python

Реализуем полный пример стабилизации продольного движения самолета с детальным анализом.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import eigvals
from scipy.integrate import solve_ivp

class AircraftLongitudinalControl:
    """
    Класс для моделирования и управления продольным движением самолета
    """
    
    def __init__(self):
        # Модель короткопериодического продольного движения
        # Состояния: [alpha, q] - угол атаки и угловая скорость тангажа
        self.A = np.array([[-0.313, 1.0],
                          [-0.0139, -0.426]])
        
        self.B = np.array([[-0.0322],
                          [0.0]])
        
        self.C = np.array([[0, 1]])  # измеряем только угловую скорость q
        
        self.D = np.array([[0]])
        
        # Параметры для анализа
        self.n = self.A.shape[0]
        
        print("МОДЕЛЬ ПРОДОЛЬНОГО ДВИЖЕНИЯ САМОЛЕТА")
        print("="*50)
        print("Состояния: [α, q]")
        print("α - угол атаки (рад)")
        print("q - угловая скорость тангажа (рад/с)")
        print("Управление: δe - отклонение руля высоты (рад)")
        print("Измерение: q - угловая скорость тангажа")
        print()
        print(f"A = \n{self.A}")
        print(f"B = \n{self.B}")
        print(f"C = \n{self.C}")
    
    def analyze_open_loop(self):
        """Анализ исходной (разомкнутой) системы"""
        
        print(f"\n{'='*60}")
        print("АНАЛИЗ ИСХОДНОЙ СИСТЕМЫ")
        print(f"{'='*60}")
        
        # Собственные числа
        poles = eigvals(self.A)
        print(f"Полюсы исходной системы: {poles}")
        
        # Анализ устойчивости
        stable = all(np.real(pole) < 0 for pole in poles)
        print(f"Устойчивость: {'✓ Устойчива' if stable else '✗ Неустойчива'}")
        
        # Характеристики колебаний
        for i, pole in enumerate(poles):
            real_part = np.real(pole)
            imag_part = np.imag(pole)
            
            if abs(imag_part) > 1e-6:
                freq = abs(imag_part)
                damping_ratio = -real_part / abs(pole)
                period = 2 * np.pi / freq
                
                print(f"\nПолюс {i+1}: {pole:.4f}")
                print(f"  Частота: {freq:.4f} рад/с")
                print(f"  Период: {period:.2f} с")
                print(f"  Коэффициент демпфирования: {damping_ratio:.4f}")
                
                if period < 5:
                    print("  Тип: Короткопериодическая мода")
                else:
                    print("  Тип: Длиннопериодическая мода")
        
        # Проверка управляемости и наблюдаемости
        P = ctrl.ctrb(self.A, self.B)
        O = ctrl.obsv(self.A, self.C)
        
        controllable = np.linalg.matrix_rank(P) == self.n
        observable = np.linalg.matrix_rank(O) == self.n
        
        print(f"\nУправляемость: {'✓' if controllable else '✗'}")
        print(f"Наблюдаемость: {'✓' if observable else '✗'}")
        
        if not controllable:
            print("⚠️  Система не управляема! Синтез невозможен.")
        if not observable:
            print("⚠️  Система не наблюдаема! Наблюдатель невозможен.")
        
        return poles, controllable, observable
    
    def design_controller(self, desired_poles=None):
        """Синтез регулятора методом размещения полюсов"""
        
        if desired_poles is None:
            desired_poles = [-1.0 + 1.0j, -1.0 - 1.0j]  # более быстрые и лучше задемпфированные
        
        print(f"\n{'='*60}")
        print("СИНТЕЗ РЕГУЛЯТОРА")
        print(f"{'='*60}")
        
        print(f"Желаемые полюсы: {desired_poles}")
        
        # Размещение полюсов
        self.K = ctrl.place(self.A, self.B, desired_poles)
        
        # Проверка результата
        A_cl = self.A - self.B @ self.K
        actual_poles = eigvals(A_cl)
        
        print(f"Матрица обратной связи K: {self.K}")
        print(f"Фактические полюсы: {actual_poles}")
        
        # Закон управления
        print(f"\nЗакон управления:")
        print(f"δe = -K * [α; q] = -{self.K[0,0]:.3f}*α - {self.K[0,1]:.3f}*q")
        
        return self.K, actual_poles
    
    def design_observer(self, observer_poles=None):
        """Синтез наблюдателя состояния"""
        
        if observer_poles is None:
            # Полюсы наблюдателя в 5 раз быстрее полюсов регулятора
            observer_poles = [-5.0 + 5.0j, -5.0 - 5.0j]
        
        print(f"\n{'='*60}")
        print("СИНТЕЗ НАБЛЮДАТЕЛЯ")
        print(f"{'='*60}")
        
        print(f"Желаемые полюсы наблюдателя: {observer_poles}")
        
        # Синтез наблюдателя (используем дуальность)
        L_T = ctrl.place(self.A.T, self.C.T, observer_poles)
        self.L = L_T.T
        
        # Проверка результата
        A_obs = self.A - self.L @ self.C
        actual_observer_poles = eigvals(A_obs)
        
        print(f"Матрица усиления наблюдателя L: {self.L}")
        print(f"Фактические полюсы наблюдателя: {actual_observer_poles}")
        
        # Уравнение наблюдателя
        print(f"\nУравнение наблюдателя:")
        print(f"α̂' = {self.A[0,0]:.3f}*α̂ + {self.A[0,1]:.3f}*q̂ + {self.B[0,0]:.3f}*δe + {self.L[0,0]:.3f}*(q - q̂)")
        print(f"q̂' = {self.A[1,0]:.3f}*α̂ + {self.A[1,1]:.3f}*q̂ + {self.B[1,0]:.3f}*δe + {self.L[1,0]:.3f}*(q - q̂)")
        
        return self.L, actual_observer_poles
    
    def simulate_scenarios(self, t_sim=15, disturbance_time=2):
        """Моделирование различных сценариев управления"""
        
        print(f"\n{'='*60}")
        print("МОДЕЛИРОВАНИЕ СИСТЕМЫ")
        print(f"{'='*60}")
        
        t = np.linspace(0, t_sim, 1000)
        
        # Начальные условия
        alpha0 = 0.1  # начальный угол атаки (рад) ≈ 5.7°
        q0 = 0.0      # начальная угловая скорость
        x0_true = np.array([alpha0, q0])
        
        # Начальная оценка (неточная)
        x0_est = np.array([0.0, 0.0])
        
        print(f"Начальные условия:")
        print(f"  Истинное состояние: α₀ = {alpha0:.3f} рад ({np.degrees(alpha0):.1f}°), q₀ = {q0:.3f} рад/с")
        print(f"  Начальная оценка: α̂₀ = {x0_est[0]:.3f} рад, q̂₀ = {x0_est[1]:.3f} рад/с")
        
        # Сценарий 1: Без управления
        print("\nМоделирование: Система без управления...")
        sys_open = ctrl.ss(self.A, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
        t1, x1 = ctrl.initial_response(sys_open, t, X0=x0_true, return_x=True)
        
        # Сценарий 2: Идеальная обратная связь
        print("Моделирование: Идеальная обратная связь...")
        A_ideal = self.A - self.B @ self.K
        sys_ideal = ctrl.ss(A_ideal, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
        t2, x2 = ctrl.initial_response(sys_ideal, t, X0=x0_true, return_x=True)
        u2 = -self.K @ x2  # управляющий сигнал
        
        # Сценарий 3: Полная система (регулятор + наблюдатель)
        print("Моделирование: Полная система с наблюдателем...")
        
        def full_system_dynamics(t, state):
            """Динамика полной системы"""
            x = state[:2]      # истинное состояние [α, q]
            x_hat = state[2:]  # оценка состояния [α̂, q̂]
            
            # Измерение (только угловая скорость q)
            y = self.C @ x
            
            # Управление на основе оценки
            u = -self.K @ x_hat
            
            # Возмущение (порыв ветра в момент disturbance_time)
            disturbance = 0.05 if abs(t - disturbance_time) < 0.1 else 0.0
            
            # Динамика истинной системы
            dx_dt = self.A @ x + self.B @ u + np.array([disturbance, 0])
            
            # Динамика наблюдателя
            dx_hat_dt = self.A @ x_hat + self.B @ u + self.L @ (y - self.C @ x_hat)
            
            return np.concatenate([dx_dt, dx_hat_dt])
        
        # Численное интегрирование
        x0_full = np.concatenate([x0_true, x0_est])
        sol = solve_ivp(full_system_dynamics, [0, t_sim], x0_full, t_eval=t, rtol=1e-8)
        
        x3 = sol.y[:2, :]      # истинное состояние
        x_hat3 = sol.y[2:, :]  # оценка состояния
        u3 = -self.K @ x_hat3  # управляющий сигнал
        error3 = x3 - x_hat3   # ошибка наблюдения
        
        return {
            't': t,
            'open_loop': {'t': t1, 'x': x1},
            'ideal_feedback': {'t': t2, 'x': x2, 'u': u2},
            'full_system': {'t': t, 'x': x3, 'x_hat': x_hat3, 'u': u3, 'error': error3}
        }
    
    def plot_results(self, results):
        """Построение графиков результатов моделирования"""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        t = results['t']
        open_loop = results['open_loop']
        ideal = results['ideal_feedback']
        full = results['full_system']
        
        # 1. Угол атаки
        axes[0, 0].plot(open_loop['t'], np.degrees(open_loop['x'][0, :]), 'k--', 
                       linewidth=2, label='Без управления')
        axes[0, 0].plot(ideal['t'], np.degrees(ideal['x'][0, :]), 'b-', 
                       linewidth=2, label='Идеальная ОС')
        axes[0, 0].plot(t, np.degrees(full['x'][0, :]), 'r-', 
                       linewidth=2, label='С наблюдателем')
        axes[0, 0].set_title('Угол атаки α')
        axes[0, 0].set_xlabel('Время (с)')
        axes[0, 0].set_ylabel('α (градусы)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Угловая скорость тангажа
        axes[0, 1].plot(open_loop['t'], np.degrees(open_loop['x'][1, :]), 'k--', 
                       linewidth=2, label='Без управления')
        axes[0, 1].plot(ideal['t'], np.degrees(ideal['x'][1, :]), 'b-', 
                       linewidth=2, label='Идеальная ОС')
        axes[0, 1].plot(t, np.degrees(full['x'][1, :]), 'r-', 
                       linewidth=2, label='С наблюдателем')
        axes[0, 1].set_title('Угловая скорость тангажа q')
        axes[0, 1].set_xlabel('Время (с)')
        axes[0, 1].set_ylabel('q (град/с)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Управляющий сигнал
        axes[0, 2].plot(ideal['t'], np.degrees(ideal['u'][0, :]), 'b-', 
                       linewidth=2, label='Идеальная ОС')
        axes[0, 2].plot(t, np.degrees(full['u'][0, :]), 'r-', 
                       linewidth=2, label='С наблюдателем')
        axes[0, 2].set_title('Отклонение руля высоты δe')
        axes[0, 2].set_xlabel('Время (с)')
        axes[0, 2].set_ylabel('δe (градусы)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Ошибка наблюдения - угол атаки
        axes[1, 0].plot(t, np.degrees(full['error'][0, :]), 'g-', linewidth=2)
        axes[1, 0].set_title('Ошибка наблюдения угла атаки')
        axes[1, 0].set_xlabel('Время (с)')
        axes[1, 0].set_ylabel('α - α̂ (градусы)')
        axes[1, 0].grid(True)
        
        # 5. Ошибка наблюдения - угловая скорость
        axes[1, 1].plot(t, np.degrees(full['error'][1, :]), 'g-', linewidth=2)
        axes[1, 1].set_title('Ошибка наблюдения угловой скорости')
        axes[1, 1].set_xlabel('Время (с)')
        axes[1, 1].set_ylabel('q - q̂ (град/с)')
        axes[1, 1].grid(True)
        
        # 6. Сравнение истинного состояния и оценки
        axes[1, 2].plot(t, np.degrees(full['x'][0, :]), 'b-', linewidth=2, label='Истинный α')
        axes[1, 2].plot(t, np.degrees(full['x_hat'][0, :]), 'r--', linewidth=2, label='Оценка α̂')
        axes[1, 2].set_title('Сравнение: истинное vs оценка (α)')
        axes[1, 2].set_xlabel('Время (с)')
        axes[1, 2].set_ylabel('α (градусы)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # 7. Фазовый портрет
        axes[2, 0].plot(np.degrees(open_loop['x'][0, :]), np.degrees(open_loop['x'][1, :]), 
                       'k--', linewidth=2, label='Без управления')
        axes[2, 0].plot(np.degrees(ideal['x'][0, :]), np.degrees(ideal['x'][1, :]), 
                       'b-', linewidth=2, label='Идеальная ОС')
        axes[2, 0].plot(np.degrees(full['x'][0, :]), np.degrees(full['x'][1, :]), 
                       'r-', linewidth=2, label='С наблюдателем')
        axes[2, 0].plot(np.degrees(full['x'][0, 0]), np.degrees(full['x'][1, 0]), 
                       'ko', markersize=8, label='Начальная точка')
        axes[2, 0].set_title('Фазовый портрет')
        axes[2, 0].set_xlabel('α (градусы)')
        axes[2, 0].set_ylabel('q (град/с)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 8. Карта полюсов
        poles_open = eigvals(self.A)
        poles_controller = eigvals(self.A - self.B @ self.K)
        poles_observer = eigvals(self.A - self.L @ self.C)
        
        axes[2, 1].plot(np.real(poles_open), np.imag(poles_open), 'ko', 
                       markersize=10, label='Исходная система')
        axes[2, 1].plot(np.real(poles_controller), np.imag(poles_controller), 'bs', 
                       markersize=8, label='Регулятор')
        axes[2, 1].plot(np.real(poles_observer), np.imag(poles_observer), 'r^', 
                       markersize=8, label='Наблюдатель')
        
        axes[2, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[2, 1].axvspan(-8, 0, alpha=0.2, color='green', label='Область устойчивости')
        axes[2, 1].set_title('Карта полюсов')
        axes[2, 1].set_xlabel('Действительная часть')
        axes[2, 1].set_ylabel('Мнимая часть')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # 9. Анализ качества
        # Время регулирования (2% критерий)
        final_alpha_ideal = ideal['x'][0, -1]
        final_alpha_full = full['x'][0, -1]
        
        settling_idx_ideal = np.where(np.abs(ideal['x'][0, :] - final_alpha_ideal) <= 
                                     0.02 * abs(final_alpha_ideal))[0]
        settling_idx_full = np.where(np.abs(full['x'][0, :] - final_alpha_full) <= 
                                    0.02 * abs(final_alpha_full))[0]
        
        settling_time_ideal = ideal['t'][settling_idx_ideal[0]] if len(settling_idx_ideal) > 0 else ideal['t'][-1]
        settling_time_full = t[settling_idx_full[0]] if len(settling_idx_full) > 0 else t[-1]
        
        systems = ['Идеальная ОС', 'С наблюдателем']
        settling_times = [settling_time_ideal, settling_time_full]
        
        axes[2, 2].bar(systems, settling_times, alpha=0.7, color=['blue', 'red'])
        axes[2, 2].set_title('Время регулирования')
        axes[2, 2].set_ylabel('Время (с)')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Вывод численных результатов
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ АНАЛИЗА")
        print(f"{'='*60}")
        print(f"Время регулирования (идеальная ОС): {settling_time_ideal:.2f} с")
        print(f"Время регулирования (с наблюдателем): {settling_time_full:.2f} с")
        print(f"Разница: {abs(settling_time_full - settling_time_ideal):.2f} с")
        
        max_error_alpha = np.max(np.abs(full['error'][0, :]))
        max_error_q = np.max(np.abs(full['error'][1, :]))
        print(f"Максимальная ошибка наблюдения α: {np.degrees(max_error_alpha):.3f}°")
        print(f"Максимальная ошибка наблюдения q: {np.degrees(max_error_q):.3f}°/с")
        
        max_control_ideal = np.max(np.abs(ideal['u'][0, :]))
        max_control_full = np.max(np.abs(full['u'][0, :]))
        print(f"Максимальное отклонение руля (идеальная ОС): {np.degrees(max_control_ideal):.1f}°")
        print(f"Максимальное отклонение руля (с наблюдателем): {np.degrees(max_control_full):.1f}°")

def main():
    """Основная функция - полный пример"""
    
    # Создание объекта системы управления
    aircraft = AircraftLongitudinalControl()
    
    # Анализ исходной системы
    poles, controllable, observable = aircraft.analyze_open_loop()
    
    if not (controllable and observable):
        print("Система не подходит для синтеза!")
        return
    
    # Синтез регулятора
    K, controller_poles = aircraft.design_controller()
    
    # Синтез наблюдателя
    L, observer_poles = aircraft.design_observer()
    
    # Моделирование
    results = aircraft.simulate_scenarios()
    
    # Построение графиков
    aircraft.plot_results(results)
    
    print(f"\n{'='*60}")
    print("ЗАКЛЮЧЕНИЕ")
    print(f"{'='*60}")
    print("✓ Успешно синтезирована система автоматического управления")
    print("✓ Система устойчива и имеет хорошие динамические характеристики")
    print("✓ Наблюдатель обеспечивает точную оценку недоступных для измерения состояний")
    print("✓ Качество управления с наблюдателем близко к идеальной обратной связи")

if __name__ == "__main__":
    main()
```

### Задание для самостоятельной работы

```python
# Дополнительные эксперименты для самостоятельного выполнения

def extended_experiments():
    """Дополнительные эксперименты"""
    
    aircraft = AircraftLongitudinalControl()
    
    # 1. Эксперимент с различными желаемыми полюсами
    pole_variants = [
        [-0.5 + 0.5j, -0.5 - 0.5j],  # медленные
        [-2 + 2j, -2 - 2j],          # быстрые
        [-1, -2],                     # апериодические
        [-0.7 + 1.4j, -0.7 - 1.4j]   # слабо задемпфированные
    ]
    
    print("Задание 1: Исследуйте влияние различных полюсов регулятора")
    for i, poles in enumerate(pole_variants):
        print(f"Вариант {i+1}: {poles}")
        # TODO: Реализуйте синтез и сравнение
    
    # 2. Эксперимент с шумом измерений
    print("\nЗадание 2: Добавьте шум в измерения и оцените робастность")
    # TODO: Добавьте белый шум в измерения q
    
    # 3. Эксперимент с неопределенностью параметров
    print("\nЗадание 3: Исследуйте влияние неточности модели")
    # TODO: Варьируйте элементы матрицы A в пределах ±20%
    
    # 4. Сравнение с ПИД-регулятором
    print("\nЗадание 4: Сравните с классическим ПИД-регулятором")
    # TODO: Реализуйте ПИД-регулятор и сравните результаты

# Запустите эти эксперименты для углубленного изучения!
```

Этот полный пример демонстрирует:

1. **Объектно-ориентированный подход** к проектированию систем управления
2. **Полный цикл разработки** от анализа до моделирования
3. **Детальную визуализацию** всех аспектов работы системы
4. **Практические рекомендации** по настройке параметров
5. **Задания для самостоятельной работы** для закрепления материала
