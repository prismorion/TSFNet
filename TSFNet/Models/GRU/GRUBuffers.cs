using TSFNet.Calculations;

namespace TSFNet.Models.GRU
{
    public class GRUBuffers
    {
        public double[][] dOutputs;
        public double[][][] inputHistory;
        public double[][][] zHistory;
        public double[][][] rHistory;
        public double[][][] hHistory;
        public double[][][] hiddenHistory;

        public double[] delta;
        public double[] prevDelta;
        public double[] prevGateDelta;
        public double[] derivative;
        public double[] qt;

        public double[] dz;
        public double[] dh;
        public double[] dr;
        public double[][] dW;
        public double[][] dU;
        public double[][] dV;
        public double[][] sumDWz;
        public double[][] sumDUz;
        public double[] sumDbz;
        public double[][] sumDWh;
        public double[][] sumDUh;
        public double[] sumDbh;
        public double[][] sumDWr;
        public double[][] sumDUr;
        public double[] sumDbr;
        public double[][] sumDV;
        public double[] sumDby;

        public L2NormAccumulator l2n;
        public double scale = 1;

        public GRUBuffers(double[] _input, double[] _hidden, double[] _output, int batchSize, int seqLen)
        {
            // буферы на каждом батче
            inputHistory = new double[batchSize][][];   // история входов
            hiddenHistory = new double[batchSize][][];  // история скрытого состояния
            zHistory = new double[batchSize][][];       // история вентиля обновления
            rHistory = new double[batchSize][][];       // история вентиля сброса
            hHistory = new double[batchSize][][];       // история кандидата
            dOutputs = new double[batchSize][];         // loss
            for (int b = 0; b < batchSize; b++)
            {
                inputHistory[b] = new double[seqLen][];
                zHistory[b] = new double[seqLen][];
                rHistory[b] = new double[seqLen][];
                hHistory[b] = new double[seqLen][];
                dOutputs[b] = new double[_output.Length];
                for (int i = 0; i < inputHistory[b].Length; i++)
                {
                    inputHistory[b][i] = new double[_input.Length];
                    zHistory[b][i] = new double[_hidden.Length];
                    rHistory[b][i] = new double[_hidden.Length];
                    hHistory[b][i] = new double[_hidden.Length];
                }

                hiddenHistory[b] = new double[seqLen + 1][];
                for (int h = 0; h < hiddenHistory[b].Length; h++)
                    hiddenHistory[b][h] = new double[_hidden.Length];
            }

            sumDV = new double[_output.Length][];
            for (int i = 0; i < _output.Length; i++)
                sumDV[i] = new double[_hidden.Length];

            sumDby = new double[_output.Length];

            dV = new double[_output.Length][];
            for (int i = 0; i < dV.Length; i++)
                dV[i] = new double[_hidden.Length];

            delta = new double[_hidden.Length];
            prevDelta = new double[_hidden.Length];
            prevGateDelta = new double[_hidden.Length];
            derivative = new double[_hidden.Length];

            dz = new double[_hidden.Length];
            dh = new double[_hidden.Length];
            dr = new double[_hidden.Length];
            qt = new double[_hidden.Length];

            dW = new double[_hidden.Length][];
            dU = new double[_hidden.Length][];
            for (int i = 0; i < _hidden.Length; i++)
            {
                dW[i] = new double[_input.Length];
                dU[i] = new double[_hidden.Length];
            }

            sumDWz = new double[_hidden.Length][];
            sumDUz = new double[_hidden.Length][];
            sumDbz = new double[_hidden.Length];
            sumDWh = new double[_hidden.Length][];
            sumDUh = new double[_hidden.Length][];
            sumDbh = new double[_hidden.Length];
            sumDWr = new double[_hidden.Length][];
            sumDUr = new double[_hidden.Length][];
            sumDbr = new double[_hidden.Length];
            for (int i = 0; i < _hidden.Length; i++)
            {
                sumDWz[i] = new double[_input.Length];
                sumDUz[i] = new double[_hidden.Length];
                sumDWh[i] = new double[_input.Length];
                sumDUh[i] = new double[_hidden.Length];
                sumDWr[i] = new double[_input.Length];
                sumDUr[i] = new double[_hidden.Length];
            }

            l2n = new L2NormAccumulator();
        }

        public void ClearAccum()
        {
            // обнуление аккумуляторов перед батчем
            for (int i = 0; i < sumDWh.Length; i++)
            {
                Array.Clear(sumDWz[i]);
                Array.Clear(sumDUz[i]);
                Array.Clear(sumDWh[i]);
                Array.Clear(sumDUh[i]);
                Array.Clear(sumDWr[i]);
                Array.Clear(sumDUr[i]);
            }
            Array.Clear(sumDbz);
            Array.Clear(sumDbh);
            Array.Clear(sumDbr);

            for (int i = 0; i < sumDV.Length; i++)
                Array.Clear(sumDV[i]);
            Array.Clear(sumDby);

            l2n.Reset();
        }
    }
}
