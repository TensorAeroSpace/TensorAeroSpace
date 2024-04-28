import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten


class Actor:
    """Модель Актера в IHDP
        Предоставляет классу Актера функцию-аппроксиматор (NN) Актера.
        Актер создает модель нейронной сети с помощью Tensorflow и может обучать сеть онлайн.
        Пользователь может выбрать количество слоев, количество нейронов, размер партии и количество эпох и активационных функций.

    Args:
        selected_inputs (_type_): Выбранные сигналы управления
        selected_states (_type_): Выбранные сигналы состояния
        tracking_states (_type_): Отслеживаемые состояния
        indices_tracking_states (_type_): Индексы отслеживаемых состояний
        number_time_steps (_type_): Количество временных шагов
        start_training (_type_): Шаг с которого начинается обучение
        layers (tuple, optional): _description_. Слои модели Defaults to (6, 1).
        activations (tuple, optional): _description_. Слои активации ('sigmoid', 'sigmoid').
        learning_rate (float, optional): скорость_обучения. Defaults to 0.9.
        learning_rate_cascaded (float, optional): Скорость обучения в каскадном режиме. Defaults to 0.9. 
        learning_rate_exponent_limit (int, optional): предел экспоненты скорости обучения. Defaults to 10.
        type_PE (str, optional): Тип PE. Defaults to '3211'.
        amplitude_3211 (int, optional): Амплитуда 3211. Defaults to 1.
        pulse_length_3211 (int, optional): Длина пульса 3211. Defaults to 15.
        WB_limits (int, optional): Лимит весов. Defaults to 30.
        maximum_input (int, optional): Максимальное значение. Defaults to 25.
        maximum_q_rate (int, optional): Максимальная скорость. Defaults to 20.
        cascaded_actor (bool, optional): Включить каскадный режим сети. Defaults to False.
        NN_initial (_type_, optional): Инициализации весов. Defaults to None.
        cascade_tracking_state (list, optional): Трекинг в каскадном режиме. Defaults to ['alpha', 'wz'].
        model_path (str, optional): Путь к модели для загрузки весов. Defaults to None.
    """
    beta_rmsprop = 0.999
    epsilon = 1e-8

    # Attributes related to the momentum
    beta_momentum = 0.9

    def __init__(self, selected_inputs, selected_states, tracking_states, indices_tracking_states,
                 number_time_steps, start_training, layers=(6, 1), activations=('sigmoid', 'sigmoid'),
                 learning_rate=0.9, learning_rate_cascaded=0.9, learning_rate_exponent_limit=10,
                 type_PE='3211', amplitude_3211=1, pulse_length_3211=15, WB_limits=30,
                 maximum_input=25, maximum_q_rate=20, cascaded_actor=False, NN_initial=None,
                 cascade_tracking_state=['alpha', 'wz'], model_path: str = None):
        
        self.number_inputs = len(selected_inputs)
        self.selected_states = selected_states
        self.cascade_tracking_state = cascade_tracking_state
        self.number_states = len(selected_states)
        self.number_tracking_states = len(tracking_states)
        self.indices_tracking_states = indices_tracking_states
        self.xt = None
        self.xt_ref = None
        self.ut = 0
        self.maximum_input = maximum_input
        self.maximum_q_rate = maximum_q_rate
        self.model_path = model_path
        # Attributes related to time
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.start_training = start_training

        # Attributes related to the NN
        self.model = None
        self.model_q = None
        if layers[-1] != 1:
            raise Exception("The last layer should have a single neuron.")
        elif len(layers) != len(activations):
            raise Exception("The number of layers needs to be equal to the number of activations.")
        self.layers = layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.learning_rate_cascaded = learning_rate_cascaded
        self.learning_rate_0 = learning_rate
        self.learning_rate_exponent_limit = learning_rate_exponent_limit
        self.WB_limits = WB_limits
        self.NN_initial = NN_initial

        # Attributes related to the persistent excitation
        self.type_PE = type_PE
        self.amplitude_3211 = amplitude_3211
        self.pulse_length_3211 = pulse_length_3211

        # Attributes related to the training of the NN
        self.dut_dWb = None
        self.dut_dWb_1 = None

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

        # Attributes related to the momentum
        self.momentum_dict = {}

        # Attributes related to RMSprop
        self.rmsprop_dict = {}

        # Declaration of the storage arrays for the weights
        self.store_weights = {}
        self.store_weights_q = {}

        # Attributes for the cascaded actor
        self.cascaded_actor = cascaded_actor
        self.dut_dq_ref = None
        self.dq_ref_dWb = None
        self.store_q = np.zeros((1, self.number_time_steps))

    def build_actor_model(self):
        """Функция, создающая сеть Actor (актера). Это полносвязная сеть. 
        Можно определить количество слоев, количество нейронов в слою, а так же функции активации
        """

        # First Neural Network
        self.model, self.store_weights = self.create_NN(self.store_weights, 120)

        # Second Neural Network for the cascaded actor
        if self.cascaded_actor:
            print("It is assumed that the input to the NNs is the tracking error.")
            tracking_states = self.cascade_tracking_state
            self.indices_tracking_states = [self.selected_states.index(tracking_states[i]) for i in
                                            range(len(tracking_states))]
            self.number_tracking_states = len(tracking_states)

            self.model_q, self.store_weights_q = self.create_NN(self.store_weights_q, 120)

        for count in range(len(self.model.trainable_variables) * 2):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0

    def save_model(self):
        """Сохранение модели
        """
        self.model.save_weights("actor_weight.h5")

    def save_dut_dWb(self):
        """Сохранение градиента
        """
        for i in range(len(self.dut_dWb)):
            np.save(f"./actor_dut_dWb/{i}_dut_dWb.txt", self.dut_dWb[i])

    def load_dut_dWb(self):
        """Загрузка градиента
        """
        line = []
        for file in glob.glob('./actor_dut_dWb/*'):
            line.append(tf.constant((np.load(file, allow_pickle=True))))
        self.dut_dWb = line
        self.dut_dWb_1 = line

    def load_model(self):
        """Загрузка весов модели
        """
        self.model.load_weights(self.model_path)

    def create_NN(self, store_weights, seed):
        """Создает NN с учетом пользовательского ввода

        Args:
            store_weights (_type_): словарь, содержащий веса и смещения
            seed (_type_): Сид, для сохранения рандомных переменных

        Returns:
            model (_type_): созданная модель NN
            store_weights (_type_): словарь, содержащий обновленные веса и смещения.
            
        """
        
        # initializer = tf.keras.initializers.GlorotNormal()
        initializer = tf.keras.initializers.VarianceScaling(
            scale=0.01, mode='fan_in', distribution='truncated_normal', seed=seed)
        model = tf.keras.Sequential()

        model.add(Flatten(input_shape=(1, 1), name='Flatten_1'))

        model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                        name='dense_1'))

        store_weights['W1'] = np.zeros((1 * self.layers[0], self.number_time_steps + 1))
        store_weights['W1'][:, self.time_step] = model.trainable_variables[0].numpy().flatten()

        for counter, layer in enumerate(self.layers[1:]):
            model.add(Dense(self.layers[counter + 1], activation=self.activations[counter + 1],
                            kernel_initializer=initializer, name='dense_' + str(counter + 2)))
            store_weights['W' + str(counter + 2)] = np.zeros(
                (self.layers[counter] * self.layers[counter + 1], self.number_time_steps + 1))
            store_weights['W' + str(counter + 2)][:, self.time_step] = model.trainable_variables[
                (counter + 1) * 2].numpy().flatten()

        return model, store_weights

    def run_actor_online(self, xt, xt_ref):
        """Сгенерируйте ввод в систему с заданным и реальным состояниями.

        Args:
            xt (_type_): текущее состояние временного шага
            xt_ref (_type_): заданное состояния текущего временного шага

        Returns:
            ut (_type_): ввод в систему и инкрементную модель
        """

        if self.cascaded_actor:
            self.xt = xt
            self.xt_ref = xt_ref

            tracked_states = np.reshape(xt[self.indices_tracking_states[0], :], [-1, 1])
            alphat_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input_alpha = tf.constant(np.array([(alphat_error)]).astype('float32'))

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                q_ref = self.model(nn_input_alpha)
            self.dq_ref_dWb = tape.gradient(q_ref, self.model.trainable_variables)

            if self.activations[-1] == 'sigmoid':
                q_ref = max(min((2 * self.maximum_q_rate * q_ref.numpy()) - self.maximum_q_rate,
                                np.reshape(self.maximum_q_rate, q_ref.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref.numpy().shape))
            elif self.activations[-1] == 'tanh':
                q_ref = max(min((self.maximum_q_rate * q_ref.numpy()),
                                np.reshape(self.maximum_q_rate, q_ref.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref.numpy().shape))

            self.store_q[:, self.time_step] = q_ref

            tracked_states_q = np.reshape(xt[self.indices_tracking_states[1], :], [-1, 1])
            qt_error = np.reshape(tracked_states_q - np.reshape(q_ref, tracked_states_q.shape), [-1, 1])
            nn_input_q = tf.constant(np.array([(qt_error)]).astype('float32'))

            with tf.GradientTape() as tape:
                tape.watch(nn_input_q)
                ut = self.model_q(nn_input_q)

            self.dut_dq_ref = tape.gradient(ut, nn_input_q)

            with tf.GradientTape() as tape:
                tape.watch(self.model_q.trainable_variables)
                ut = self.model_q(nn_input_q)

            self.dut_dWb = tape.gradient(ut, self.model_q.trainable_variables)

        else:
            self.xt = xt
            self.xt_ref = xt_ref

            tracked_states = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = tf.constant(np.array([(xt_error)]).astype('float32'))

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                ut = self.model(nn_input)
            self.dut_dWb = tape.gradient(ut, self.model.trainable_variables)

        e0 = self.compute_persistent_excitation()

        if self.activations[-1] == 'sigmoid':
            self.ut = max(min((2 * self.maximum_input * ut.numpy()) - self.maximum_input + e0,
                              np.reshape(self.maximum_input, ut.numpy().shape)),
                          np.reshape(-self.maximum_input, ut.numpy().shape))
        elif self.activations[-1] == 'tanh':
            ut = max(min((self.maximum_input * ut.numpy()),
                         np.reshape(self.maximum_input, ut.numpy().shape)),
                     np.reshape(-self.maximum_input, ut.numpy().shape))

            self.ut = max(min(ut + e0,
                              np.reshape(self.maximum_input, ut.shape)),
                          np.reshape(-self.maximum_input, ut.shape))

        return self.ut

    def train_actor_online(self, Jt1, dJt1_dxt1, G):
        """Получает элементы цепного правила, вычисляет градиент и применяет его к соответствующим весам и смещениям.

        Args:
            Jt1 (_type_): dEa/dJ
            dJt1_dxt1 (_type_): dJ/dx
            G (_type_): dx/du, полученное из инкрементной модели
        """
        
        Jt1 = Jt1.flatten()[0]

        chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states, :], [-1, 1]).T, dJt1_dxt1)

        chain_rule = chain_rule.flatten()[0]
        for count in range(len(self.dut_dWb)):
            update = chain_rule * self.dut_dWb[count]
            self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update,
                                                                        self.model.trainable_variables[count].shape))

            # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
            self.model = self.check_WB_limits(count, self.model)

    def train_actor_online_adaptive_alpha(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """Обучение Actor (актера). с помощью адаптивной альфы в зависимости от знака и величины сетевых ошибок

        Args:
            Jt1 (_type_): оценка критика с предсказанием следующего временного шага инкрементной модели
            dJt1_dxt1 (_type_): градиент критической сети по отношению к следующему прогнозу времени инкрементной модели
            G (_type_): матрица распределения входных данных
            incremental_model (_type_): инкрементная модель
            critic (_type_): Критик
            xt_ref1 (_type_): заданное состояния на следующем временном шаге
        """
        Ec_actor_before = 0.5 * np.square(Jt1)
        # print("ACTOR LOSS xt1 before= ", Ec_actor_before)
        weight_cache = [tf.Variable(self.model.trainable_variables[i].numpy()) for i in
                        range(len(self.model.trainable_variables))]
        network_improvement = False
        n_reductions = 0
        while not network_improvement and self.time_step > self.start_training:
            # Train the actor
            self.train_actor_online(Jt1, dJt1_dxt1, G)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)

            # Code for checking whether the learning rate of the actor should be halved
            if Ec_actor_after <= Ec_actor_before or n_reductions > 10:
                network_improvement = True
                if np.sign(Jt1) == np.sign(Jt1_after):
                    self.learning_rate = min(2 * self.learning_rate,
                                             self.learning_rate_0 * 2 ** self.learning_rate_exponent_limit)
                    # print("ACTOR LEARNING_RATE = ", self.learning_rate)
            else:
                n_reductions += 1
                self.learning_rate = max(self.learning_rate / 2,
                                         self.learning_rate_0 / 2 ** self.learning_rate_exponent_limit)
                for WB_count in range(len(self.model.trainable_variables)):
                    self.model.trainable_variables[WB_count].assign(weight_cache[WB_count].numpy())
                # print("ACTOR LEARNING_RATE = ", self.learning_rate)

    def train_actor_online_adam(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """Обучение Actor (актера) с помощью оптимизатора Adam.

        Args:
            Jt1 (_type_): оценка критика с предсказанием следующего временного шага инкрементной модели
            dJt1_dxt1 (_type_): градиент критической сети по отношению к следующему прогнозу времени инкрементной модели
            G (_type_): матрица распределения входных данных
            incremental_model (_type_): инкрементная модель
            critic (_type_): Критик
            xt_ref1 (_type_): заданное состояния на следующем временном шаге
        """
        if self.cascaded_actor:
            # Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training and np.abs(self.ut) < 25:
                for count in range(len(self.dut_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = 2 * self.maximum_input * chain_rule * self.dut_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = self.maximum_input * chain_rule * self.dut_dWb[count]
                    else:
                        raise Exception("There is no code for the defined output activation function.")
                    self.model_q, self.learning_rate_cascaded = self.compute_Adam_update(count, gradient,
                                                                                         self.model_q,
                                                                                         self.learning_rate_cascaded)

                for count in range(len(self.dq_ref_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = -2 * self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = -self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    else:
                        raise Exception("There is no code for the defined output activation function.")

                    self.model, self.learning_rate = self.compute_Adam_update(count, gradient,
                                                                              self.model, self.learning_rate)
            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            #Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)
        else:
            #Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training:
                for count in range(len(self.dut_dWb)):
                    gradient = chain_rule * self.dut_dWb[count]
                    self.model, self.learning_rate = self.compute_Adam_update(count, gradient,
                                                                              self.model, self.learning_rate)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            # Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)

    def train_actor_online_alpha_decay(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """Обучите актера со скоростью обучения, которая уменьшается с количеством временных шагов

        Args:
            Jt1 (_type_): оценка критика с предсказанием следующего временного шага инкрементной модели
            dJt1_dxt1 (_type_): градиент критической сети по отношению к следующему прогнозу времени инкрементной модели
            G (_type_): матрица распределения входных данных
            incremental_model (_type_): инкрементная модель
            critic (_type_): Критик
            xt_ref1 (_type_): заданное состояния на следующем временном шаге
        """
        
        if self.cascaded_actor:
            # Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training and np.abs(self.ut) < 25:
                for count in range(len(self.dut_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = 2 * self.maximum_input * chain_rule * self.dut_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = self.maximum_input * chain_rule * self.dut_dWb[count]
                    self.model_q.trainable_variables[count].assign_sub(
                        np.reshape(self.learning_rate_cascaded * gradient,
                                   self.model_q.trainable_variables[
                                       count].shape))

                    # Implement WB_limits: the weights and biases can not have values whose absolute value
                    # exceeds WB_limits
                    self.model_q = self.check_WB_limits(count, self.model_q)
                    if count % 2 == 1:
                        self.model_q.trainable_variables[count].assign(
                            np.zeros(self.model_q.trainable_variables[count].shape))
                for count in range(len(self.dq_ref_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = -2 * self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = -self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * gradient,
                                                                                self.model.trainable_variables[
                                                                                    count].shape))
                    self.model = self.check_WB_limits(count, self.model)
                    if count % 2 == 1:
                        self.model.trainable_variables[count].assign(
                            np.zeros(self.model.trainable_variables[count].shape))

                # Update the learning rate
                self.learning_rate = max(self.learning_rate * 0.9995, 0.0001)
                self.learning_rate_cascaded = max(self.learning_rate_cascaded * 0.9995, 0.0001)
            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            # incremental_model.identify_incremental_model_LS(self.xt, ut_after)
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            # Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)
        else:
            # Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training:
                for count in range(len(self.dut_dWb)):
                    gradient = chain_rule * self.dut_dWb[count]
                    self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * gradient,
                                                                                self.model.trainable_variables[
                                                                                    count].shape))
                    # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                    self.model = self.check_WB_limits(count, self.model)
                    if count % 2 == 1:
                        self.model.trainable_variables[count].assign(
                            np.zeros(self.model.trainable_variables[count].shape))
                # Update the learning rate
                self.learning_rate = max(self.learning_rate * 0.995, 0.001)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            # Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)

    def compute_Adam_update(self, count, gradient, model, learning_rate):
        """Вычисляет обновление Adam и применяет его к обновлениям веса.

        Args:
            count (_type_): индекс анализируемой весовой матрицы
            gradient (_type_): вычисленный градиент для каждого из весов
            model (_type_): модель где обновляем веса
            learning_rate (_type_): скорость обучения анализируемой модели

        Returns:
            model (_type_): обновленная модель
            learning_rate (_type_): обновленная скорость обучения
        """
        
        momentum = self.beta_momentum * self.momentum_dict[count] + (1 - self.beta_momentum) * gradient
        self.momentum_dict[count] = momentum
        momentum_corrected = momentum / (1 - np.power(self.beta_momentum, self.time_step + 1))

        rmsprop = self.beta_rmsprop * self.rmsprop_dict[count] + \
                  (1 - self.beta_rmsprop) * np.multiply(gradient, gradient)
        self.rmsprop_dict[count] = rmsprop
        rmsprop_corrected = rmsprop / (1 - np.power(self.beta_rmsprop, self.time_step + 1))

        update = momentum_corrected / (np.sqrt(rmsprop_corrected) + self.epsilon)
        model.trainable_variables[count].assign_sub(np.reshape(learning_rate * update,
                                                               model.trainable_variables[count].shape))
        # Implement WB_limits: the weights and biases can not have values whose absolute value
        # exceeds WB_limits
        model = self.check_WB_limits(count, model)
        if count % 2 == 1:
            model.trainable_variables[count].assign(np.zeros(model.trainable_variables[count].shape))

        if count == len(model.trainable_variables) - 1:
            learning_rate = max(learning_rate * 0.9995, 0.0001)

        return model, learning_rate

    def check_WB_limits(self, count, model):
        """Проверка, не превышают ли какие-либо веса и смещения установленный предел (WB_limits), и насыщайте значения.

        Args:
            count (_type_): индекс в анализируемой модели model.trainable_variables
            model (_type_): модель актера

        Returns:
            _type_: модель актера
        """
        
        WB_variable = model.trainable_variables[count].numpy()
        WB_variable[WB_variable > self.WB_limits] = self.WB_limits
        WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
        model.trainable_variables[count].assign(WB_variable)
        return model

    def compute_persistent_excitation(self, *args):
        """Расчет постоянного возбуждения на каждом временном шаге. 

        Returns:
            e0 (_type_): отклонение PE
        """
        if len(args) == 1:
            t = args[0] + 1
        elif len(args) == 0:
            t = self.time_step + 1

        e0_1 = 0
        e0_2 = 0
        if self.type_PE == 'sinusoidal' or self.type_PE == 'combined':
            e0_1 = np.sin(t) * np.cos(2 * t) * (np.sin(3 * t + np.pi / 4) + np.cos(4 * t - np.pi / 3)) * 1e-2

        if self.type_PE == '3211' or self.type_PE == 'combined':
            if t < 3 * self.pulse_length_3211 / 7:
                e0_2 = 0.5 * self.amplitude_3211
            elif t < 5 * self.pulse_length_3211 / 7:
                e0_2 = -0.5 * self.amplitude_3211
            elif t < 6 * self.pulse_length_3211 / 7:
                e0_2 = 0.8 * self.amplitude_3211
            elif t < self.pulse_length_3211:
                e0_2 = -self.amplitude_3211

        e0 = e0_1 + e0_2

        return e0

    def update_actor_attributes(self):
        """Атрибуты, которые меняются с каждым временным шагом, обновляются
        """
        self.time_step += 1
        self.dut_dWb_1 = self.dut_dWb

        for counter in range(len(self.layers)):
            self.store_weights['W' + str(counter + 1)][:, self.time_step] = self.model.trainable_variables[
                counter * 2].numpy().flatten()

        if self.cascaded_actor:
            for counter in range(len(self.layers)):
                self.store_weights_q['W' + str(counter + 1)][:, self.time_step] = self.model_q.trainable_variables[
                    counter * 2].numpy().flatten()

    def evaluate_actor(self, *args):
        """Оценка актора с учетом входных данных или атрибутов, хранящихся в объекте
        
        Args:
            args: реальное и эталонное состояния могут быть предоставлены в качестве входных данных для оценки или нет, если они уже сохранены
        
        Raises:
            Exception: _description_

        Returns:
            ut _type_: ввод в систему и инкрементную модель
        """
        if len(args) == 0:
            xt = self.xt
            xt_ref = self.xt_ref
        elif len(args) == 1:
            xt = self.xt
            xt_ref = self.xt_ref
            time_step = args[0]
        elif len(args) == 2:
            xt = args[0]
            xt_ref = args[1]
        else:
            raise Exception("THERE SHOULD BE AN OUTPUT in the evaluate_actor function.")

        if self.cascaded_actor:
            tracked_states = np.reshape(xt[self.indices_tracking_states[0], :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = tf.constant(np.array([(xt_error)]).astype('float32'))

            q_ref_0 = self.model(nn_input)
            if self.activations[-1] == 'sigmoid':
                q_ref = max(min((2 * self.maximum_q_rate * q_ref_0.numpy()) - self.maximum_q_rate,
                                np.reshape(self.maximum_q_rate, q_ref_0.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref_0.numpy().shape))
            elif self.activations[-1] == 'tanh':
                q_ref = max(min((self.maximum_q_rate * q_ref_0.numpy()),
                                np.reshape(self.maximum_q_rate, q_ref_0.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref_0.numpy().shape))

            tracked_states = np.reshape(xt[self.indices_tracking_states[1], :], [-1, 1])
            xt_error_q = np.reshape(tracked_states - np.reshape(q_ref, tracked_states.shape), [-1, 1])
            nn_input_q = tf.constant(np.array([(xt_error_q)]).astype('float32'))

            ut = self.model_q(nn_input_q).numpy()

        else:
            tracked_states = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = tf.constant(np.array([(xt_error)]).astype('float32'))

            ut = self.model(nn_input).numpy()

        if len(args) == 1:
            e0 = self.compute_persistent_excitation(time_step)
        else:
            e0 = self.compute_persistent_excitation()

        if self.activations[-1] == 'sigmoid':
            ut = max(min((2 * self.maximum_input * ut) - self.maximum_input + e0,
                         np.reshape(self.maximum_input, ut.shape)),
                     np.reshape(-self.maximum_input, ut.shape))
        elif self.activations[-1] == 'tanh':
            ut = max(min(((self.maximum_input + 10) * ut) + e0,
                         np.reshape(self.maximum_input, ut.shape)),
                     np.reshape(-self.maximum_input, ut.shape))
        return ut
    
    def restart_time_step(self):
        """Перезапуск временного шага
        """
        self.time_step = 0

    def restart_actor(self):
        """
        Перезапустите атрибуты актера
        """
        self.time_step = 0
        self.xt = None
        self.xt_ref = None
        self.ut = 0

        # Attributes related to the training of the NN
        self.dut_dWb = None
        self.dut_dWb_1 = None
        self.learning_rate = self.learning_rate_0

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

        # Restart momentum and rmsprop
        for count in range(len(self.model.trainable_variables)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0
