using TSFNet.Calculations;

namespace TSFNet.Models.RNN
{
    public class RNNBuffers
    {
        public double[][] dOutputs;
        public double[][][] hiddenHistory;
        public double[][][] inputHistory;

        public double[][] dW;
        public double[][] dU;
        public double[][] dV;
        public double[][] sumDW;
        public double[][] sumDU;
        public double[][] sumDV;
        public double[] sumDbh;
        public double[] sumDby;

        public double[] delta;
        public double[] prevDelta;
        public double[] derivative;

        public L2NormAccumulator l2n;
        public double scale = 1;

        public RNNBuffers(double[] _input, double[] _hidden, double[] _output, int batchSize, int seqLen)
        {
            // буферы на каждом батче
            dOutputs = new double[batchSize][];         // для хранения loss
            hiddenHistory = new double[batchSize][][];  // для клонирования слоёв скрытого состояния
            inputHistory = new double[batchSize][][];   // для клонирования слоёв входа
            for (int b = 0; b < batchSize; b++)
            {
                dOutputs[b] = new double[_output.Length];

                hiddenHistory[b] = new double[seqLen + 1][];
                for (int h = 0; h < hiddenHistory[b].Length; h++)
                    hiddenHistory[b][h] = new double[_hidden.Length];

                inputHistory[b] = new double[seqLen][];
                for (int i = 0; i < inputHistory[b].Length; i++)
                    inputHistory[b][i] = new double[_input.Length];
            }

            dW = new double[_hidden.Length][];
            dU = new double[_hidden.Length][];
            for (int i = 0; i < _hidden.Length; i++)
            {
                dW[i] = new double[_input.Length];
                dU[i] = new double[_hidden.Length];
            }

            dV = new double[_output.Length][];
            for (int i = 0; i < _output.Length; i++)
                dV[i] = new double[_hidden.Length];

            // аккумуляторы градиента по батчу
            sumDW = new double[_hidden.Length][];
            sumDU = new double[_hidden.Length][];
            sumDbh = new double[_hidden.Length];
            for (int i = 0; i < _hidden.Length; i++)
            {
                sumDW[i] = new double[_input.Length];
                sumDU[i] = new double[_hidden.Length];
            }

            sumDV = new double[_output.Length][];
            sumDby = new double[_output.Length];
            for (int i = 0; i < _output.Length; i++)
                sumDV[i] = new double[_hidden.Length];

            delta = new double[_hidden.Length];
            prevDelta = new double[_hidden.Length];
            derivative = new double[_hidden.Length];

            l2n = new L2NormAccumulator();
        }

        public void ClearAccum()
        {
            // обнуление аккумуляторов перед батчем
            for (int i = 0; i < sumDW.Length; i++)
            {
                Array.Clear(sumDW[i]);
                Array.Clear(sumDU[i]);
            }

            for (int i = 0; i < sumDV.Length; i++)
                Array.Clear(sumDV[i]);

            Array.Clear(sumDbh);
            Array.Clear(sumDby);

            l2n.Reset();
        }
    }
}
