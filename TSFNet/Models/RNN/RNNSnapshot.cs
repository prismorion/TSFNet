namespace TSFNet.Models.RNN
{
    public class RNNSnapshot
    {
        // веса
        public double[][] W;
        public double[][] U;
        public double[][] V;

        // сдвиги
        public double[] bh;
        public double[] by;

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
