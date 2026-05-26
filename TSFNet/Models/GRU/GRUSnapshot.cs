using TSFNet.Calculations;

namespace TSFNet.Models.GRU
{
    public class GRUSnapshot
    {
        // веса
        public double[][] Wz;
        public double[][] Uz;
        public double[] bz;

        public double[][] Wr;
        public double[][] Ur;
        public double[] br;

        public double[][] Wh;
        public double[][] Uh;
        public double[] bh;

        public double[][] V;
        public double[] by;

        public GRUSnapshot(int inputSize, int hiddenSize, int outputSize)
        {
            Wz = new double[hiddenSize][];
            Wr = new double[hiddenSize][];
            Wh = new double[hiddenSize][];
            for (int i = 0; i < hiddenSize; i++)
            {
                Wz[i] = new double[inputSize];
                Wr[i] = new double[inputSize];
                Wh[i] = new double[inputSize];
            }

            Uz = new double[hiddenSize][];
            Ur = new double[hiddenSize][];
            Uh = new double[hiddenSize][];
            for (int i = 0; i < hiddenSize; i++)
            {
                Uz[i] = new double[hiddenSize];
                Ur[i] = new double[hiddenSize];
                Uh[i] = new double[hiddenSize];
            }

            bz = new double[hiddenSize];
            br = new double[hiddenSize];
            bh = new double[hiddenSize];

            V = new double[outputSize][];
            for (int i = 0; i < outputSize; i++)
                V[i] = new double[hiddenSize];

            by = new double[outputSize];
        }
    }
}
