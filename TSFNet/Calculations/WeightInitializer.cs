namespace TSFNet.Calculations
{
    /// <summary>
    /// Предоставляет методы инициализации весов нейронной сети.
    /// </summary>
    internal static class WeightInitializer
    {
        private readonly static Random rand = new Random();

        /// <summary>
        /// Инициализация весов методом Xavier / Glorot.
        /// Подходит для сигмоидных функций активации (tanh, sigmoid).
        /// </summary>
        /// <param name="weights">Матрица весов слоя.</param>
        /// <param name="fanIn">Количество входных нейронов (откуда идут связи).</param>
        /// <param name="fanOut">Количество выходных нейронов (куда идут связи).</param>
        public static void Xavier(double[][] weights, int fanIn, int fanOut)
        {
            double a = Math.Sqrt(6.0 / (fanIn + fanOut));

            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].Length; j++)
                    weights[i][j] = (rand.NextDouble() * 2 - 1) * a;
        }

        /// <summary>
        /// Инициализация весов методом Хе (Kaiming).
        /// Подходит для функции активации семейства ReLU.
        /// </summary>
        /// <param name="weights">Матрица весов слоя.</param>
        /// <param name="fanIn">Количество входных нейронов (откуда идут связи).</param>
        /// <param name="negativeSlope">Коэффициент наклона ReLU на отрицательной части.</param>
        public static void He(double[][] weights, int fanIn, double negativeSlope = 0)
        {
            double a = Math.Sqrt(6.0 / ((1 + Math.Pow(negativeSlope, 2)) * fanIn));

            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].Length; j++)
                    weights[i][j] = (rand.NextDouble() * 2 - 1) * a;
        }
    }
}
