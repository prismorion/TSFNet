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
        public Hyperparameters(double _learningRate = 0.01, int _batchSize = 1, double _l2Lambda = 0, double _threshold = 5)
        {
            learningRate = _learningRate;
            batchSize = _batchSize;
            l2Lambda = _l2Lambda;
            threshold = _threshold;
        }
    }
}
