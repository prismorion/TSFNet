namespace TSFNet.Calculations
{
    /// <summary>
    /// Функции потерь и их производные для обучения сети.
    /// </summary>
    public static class LossFunctions
    {
        /// <summary>
        /// Среднеквадратичная ошибка (MSE) между истинным и предсказанным вектором.
        /// </summary>
        public static double MSE(double[] yTrue, double[] yPred)
        {
            int n = yTrue.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double d = yTrue[i] - yPred[i];
                sum += d * d;
            }
            return sum / (2 * n);
        }

        /// <summary>
        /// Градиент MSE по вектору предсказания.
        /// </summary>
        public static void MSEDerivative(double[] yTrue, double[] yPred, double[] dest)
        {
            int n = yTrue.Length;
            for (int i = 0; i < n; i++)
                dest[i] = (yPred[i] - yTrue[i]) / n;
        }
    }
}
