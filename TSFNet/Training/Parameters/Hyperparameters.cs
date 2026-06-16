namespace TSFNet.Training.Parameters
{
    /// <summary>
    /// Гиперпараметры алгоритма обучения: скорость обучения, размер батча,
    /// L2-регуляризация и порог отсечения градиента.
    /// </summary>
    public class Hyperparameters
    {
        /// <summary> Скорость обучения (шаг градиентного спуска). </summary>
        public double learningRate { get; set; }

        /// <summary> Размер батча. </summary>
        public int batchSize { get; set; }

        /// <summary> Коэффициент L2-регуляризации (0 — регуляризация выключена). </summary>
        public double l2Lambda { get; set; }

        /// <summary> Порог L2-нормы для отсечения градиента. </summary>
        public double threshold { get; set; }

        /// <summary>
        /// Создание набора гиперпараметров со значениями по умолчанию.
        /// </summary>
        public Hyperparameters(double learningRate = 0.01, int batchSize = 1, double l2Lambda = 0, double threshold = 5)
        {
            this.learningRate = learningRate;
            this.batchSize = batchSize;
            this.l2Lambda = l2Lambda;
            this.threshold = threshold;
        }
    }
}
