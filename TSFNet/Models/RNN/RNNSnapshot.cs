namespace TSFNet.Models.RNN
{
    /// <summary>
    /// Снимок параметров RNN (веса W, U, V и сдвиги) для сохранения и восстановления лучшего состояния сети.
    /// </summary>
    public class RNNSnapshot
    {
        // веса
        public double[][] W;
        public double[][] U;
        public double[][] V;

        // сдвиги
        public double[] bh;
        public double[] by;

        /// <summary>
        /// Выделение массивов весов и сдвигов под размеры сети.
        /// </summary>
        public RNNSnapshot(int inputSize, int hiddenSize, int outputSize)
        {
            W = new double[hiddenSize][];
            for (int i = 0; i < hiddenSize; i++)
                W[i] = new double[inputSize];

            U = new double[hiddenSize][];
            for (int i = 0; i < hiddenSize; i++)
                U[i] = new double[hiddenSize];

            V = new double[outputSize][];
            for (int i = 0; i < outputSize; i++)
                V[i] = new double[hiddenSize];

            bh = new double[hiddenSize];
            by = new double[outputSize];
        }
    }
}
