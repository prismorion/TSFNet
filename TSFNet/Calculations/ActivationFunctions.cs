namespace TSFNet.Calculations
{
    /// <summary>
    /// Функции активации и их производные для нейронных сетей.
    /// Производные sigmoid, tanh и Leaky ReLU принимают выход активации, а не предактивацию.
    /// </summary>
    internal static class ActivationFunctions
    {
        /// <summary>
        /// Сигмоида: σ(x) = 1 / (1 + e⁻ˣ). Диапазон (0, 1).
        /// </summary>
        public static void Sigmoid(double[] x, double[] dest)
        {
            for (int i = 0; i < x.Length; i++)
                dest[i] = 1 / (1 + Math.Exp(-x[i]));
        }

        /// <summary>
        /// Производная сигмоиды: σ'= σ · (1 − σ). Принимает выход Sigmoid.
        /// </summary>
        public static void SigmoidDerivative(double[] sigmoidOutput, double[] dest)
        {
            for (int i = 0; i < sigmoidOutput.Length; i++)
                dest[i] = sigmoidOutput[i] * (1 - sigmoidOutput[i]);
        }

        /// <summary>
        /// Гиперболический тангенс: tanh(x). Диапазон (-1, 1).
        /// </summary>
        public static void Tanh(double[] x, double[] dest)
        {
            for (int i = 0; i < x.Length; i++)
                dest[i] = Math.Tanh(x[i]);
        }

        /// <summary>
        /// Производная tanh: 1 − t². Принимает выход Tanh.
        /// </summary>
        public static void TanhDerivative(double[] tanhOutput, double[] dest)
        {
            for (int i = 0; i < tanhOutput.Length; i++)
                dest[i] = 1 - tanhOutput[i] * tanhOutput[i];
        }

        /// <summary>
        /// Leaky ReLU: x при x > 0, иначе 0.01·x.
        /// </summary>
        public static void LReLU(double[] x, double[] dest)
        {
            for (int i = 0; i < x.Length; i++)
                dest[i] = x[i] > 0 ? x[i] : 0.01 * x[i];
        }

        /// <summary>
        /// Производная Leaky ReLU: 1 при выходе > 0, иначе 0.01. Принимает выход LReLU.
        /// </summary>
        public static void LReLUDerivative(double[] x, double[] dest)
        {
            for (int i = 0; i < x.Length; i++)
                dest[i] = x[i] > 0 ? 1 : 0.01;
        }
    }
}
