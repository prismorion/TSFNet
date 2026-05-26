using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Training;
using TSFNet.Training.Parameters;

namespace TSFNet.Models.RNN
{
    public class RNN : ITrainable<double[][], RNNBuffers, RNNSnapshot>
    {
        // слои нейронов
        private double[] _input;
        private double[] _hidden;
        private double[] _output;

        // веса
        private double[][] W; // от input к hidden(t)
        private double[][] U; // от hidden(t-1) к hidden(t)
        private double[][] V; // от hidden(t) к output

        // сдвиги
        private double[] bh; // сдвиг hidden(t)
        private double[] by; // сдвиг output

        public RNN(int inputSize, int hiddenSize, int outputSize)
        {
            // инициализация массивов
            _input = new double[inputSize];
            _hidden = new double[hiddenSize];
            _output = new double[outputSize];

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

            // инициализация весов
            WeightInitializer.Xavier(W, inputSize, hiddenSize);
            WeightInitializer.Xavier(U, hiddenSize, hiddenSize);
            WeightInitializer.Xavier(V, hiddenSize, outputSize);
        }

        public double[] Forward(double[] input)
        {
            Array.Copy(input, _input, input.Length);
            double[] temp = new double[_hidden.Length];

            LinAlgMath.MulMatVec(U, _hidden, temp);
            LinAlgMath.MulMatVec(W, _input, _hidden);
            LinAlgMath.AddVec(_hidden, temp, _hidden);
            LinAlgMath.AddVec(_hidden, bh, _hidden);
            ActivationFunctions.Tanh(_hidden, _hidden);

            LinAlgMath.MulMatVec(V, _hidden, _output);
            LinAlgMath.AddVec(_output, by, _output);

            return (double[])_output.Clone();
        }

        public double[] Forward(double[][] input)
        {
            ClearHiddenState();
            for (int i = 0; i < input.Length; i++)
                Forward(input[i]);

            return (double[])_output.Clone();
        }

        public void ClearHiddenState()
        {
            Array.Clear(_hidden);
        }        

        public void Train(Dataset<double[][]> dataset, Hyperparameters options, RNNBuffers rnnBuffers)
        {
            // проход по примерам с шагом в размер батча
            for (int i = 0; i < dataset.Length; i += options.batchSize)
            {
                // выбор длины батча: batchSize или оставшийся хвост
                int size = Math.Min(options.batchSize, dataset.Length - i);

                // проход по батчу
                for (int b = 0; b < size; b++)
                {
                    // очистка скрытого состояния
                    ClearHiddenState();

                    // проход по таймстепам
                    for (int t = 0; t < dataset.GetInput(i + b).Length; t++)
                    {
                        Array.Copy(dataset.GetInput(i + b)[t], rnnBuffers.inputHistory[b][t], _input.Length);
                        Forward(dataset.GetInput(i + b)[t]);
                        Array.Copy(_hidden, rnnBuffers.hiddenHistory[b][t + 1], _hidden.Length);
                    }

                    LossFunctions.MSEDerivative(dataset.GetTarget(i + b), _output, rnnBuffers.dOutputs[b]);
                }

                Backward(rnnBuffers, size, options.learningRate, options.l2Lambda, options.threshold);
            }
        }

        private void Backward(RNNBuffers rnnBuffers, int size, double learningRate, double l2Lambda, double threshold)
        {
            int sequenceLength = rnnBuffers.hiddenHistory[0].Length;

            rnnBuffers.ClearAccum();

            // проход по примерам батча
            for (int b = 0; b < size; b++)
            {
                // градиент по V и by
                LinAlgMath.MulOuterVec(rnnBuffers.dOutputs[b], rnnBuffers.hiddenHistory[b][sequenceLength - 1], rnnBuffers.dV);
                LinAlgMath.AddMat(rnnBuffers.sumDV, rnnBuffers.dV, rnnBuffers.sumDV);
                LinAlgMath.AddVec(rnnBuffers.sumDby, rnnBuffers.dOutputs[b], rnnBuffers.sumDby);

                // прокидываем градиент на предыдущий слой
                LinAlgMath.MulMatTVec(V, rnnBuffers.dOutputs[b], rnnBuffers.delta);
                ActivationFunctions.TanhDerivative(rnnBuffers.hiddenHistory[b][sequenceLength - 1], rnnBuffers.derivative);
                LinAlgMath.MulElemVec(rnnBuffers.delta, rnnBuffers.derivative, rnnBuffers.delta);

                // проход по развёртке
                for (int sl = sequenceLength - 2; sl >= 0; sl--)
                {
                    LinAlgMath.MulOuterVec(rnnBuffers.delta, rnnBuffers.hiddenHistory[b][sl], rnnBuffers.dU);
                    LinAlgMath.AddMat(rnnBuffers.sumDU, rnnBuffers.dU, rnnBuffers.sumDU);

                    LinAlgMath.MulOuterVec(rnnBuffers.delta, rnnBuffers.inputHistory[b][sl], rnnBuffers.dW);
                    LinAlgMath.AddMat(rnnBuffers.sumDW, rnnBuffers.dW, rnnBuffers.sumDW);

                    LinAlgMath.AddVec(rnnBuffers.sumDbh, rnnBuffers.delta, rnnBuffers.sumDbh);

                    // если есть ещё развёртка - прокидываем градиент на предыдущий слой
                    if (sl > 0)
                    {
                        LinAlgMath.MulMatTVec(U, rnnBuffers.delta, rnnBuffers.prevDelta);
                        ActivationFunctions.TanhDerivative(rnnBuffers.hiddenHistory[b][sl], rnnBuffers.derivative);
                        LinAlgMath.MulElemVec(rnnBuffers.prevDelta, rnnBuffers.derivative, rnnBuffers.delta);
                    }
                }
            }

            // апдейт весов
            if (l2Lambda > 0)
            {
                LinAlgMath.MulMatNum(W, 1.0 - learningRate * l2Lambda, W);
                LinAlgMath.MulMatNum(U, 1.0 - learningRate * l2Lambda, U);
                LinAlgMath.MulMatNum(V, 1.0 - learningRate * l2Lambda, V);
            }

            rnnBuffers.l2n.AddMatrices(rnnBuffers.sumDW, rnnBuffers.sumDU, rnnBuffers.sumDV);
            rnnBuffers.l2n.AddVectors(rnnBuffers.sumDbh, rnnBuffers.sumDby);
            rnnBuffers.scale = Math.Min(1.0, threshold / rnnBuffers.l2n.GetNorm());

            LinAlgMath.MulMatNum(rnnBuffers.sumDW, rnnBuffers.scale * learningRate / size, rnnBuffers.sumDW);
            LinAlgMath.SubMat(W, rnnBuffers.sumDW, W);

            LinAlgMath.MulMatNum(rnnBuffers.sumDU, rnnBuffers.scale * learningRate / size, rnnBuffers.sumDU);
            LinAlgMath.SubMat(U, rnnBuffers.sumDU, U);

            LinAlgMath.MulMatNum(rnnBuffers.sumDV, rnnBuffers.scale * learningRate / size, rnnBuffers.sumDV);
            LinAlgMath.SubMat(V, rnnBuffers.sumDV, V);

            LinAlgMath.MulVecNum(rnnBuffers.sumDbh, rnnBuffers.scale * learningRate / size, rnnBuffers.sumDbh);
            LinAlgMath.SubVec(bh, rnnBuffers.sumDbh, bh);

            LinAlgMath.MulVecNum(rnnBuffers.sumDby, rnnBuffers.scale * learningRate / size, rnnBuffers.sumDby);
            LinAlgMath.SubVec(by, rnnBuffers.sumDby, by);
        }

        public RNNBuffers CreateBuffer(Dataset<double[][]> dataset, Hyperparameters options)
            => new RNNBuffers(_input, _hidden, _output, options.batchSize, dataset.GetRawInput[0].Length);

        public RNNSnapshot CreateSnapshotBuffer()
            => new RNNSnapshot(_input.Length, _hidden.Length, _output.Length);

        public void SaveSnapshot(RNNSnapshot rnnSnapshot)
        {
            ArrayCopyUtils.Copy(W, rnnSnapshot.W);
            ArrayCopyUtils.Copy(U, rnnSnapshot.U);
            ArrayCopyUtils.Copy(V, rnnSnapshot.V);

            ArrayCopyUtils.Copy(bh, rnnSnapshot.bh);
            ArrayCopyUtils.Copy(by, rnnSnapshot.by);
        }

        public void RestoreSnapshot(RNNSnapshot rnnSnapshot)
        {
            ArrayCopyUtils.Copy(rnnSnapshot.W, W);
            ArrayCopyUtils.Copy(rnnSnapshot.U, U);
            ArrayCopyUtils.Copy(rnnSnapshot.V, V);

            ArrayCopyUtils.Copy(rnnSnapshot.bh, bh);
            ArrayCopyUtils.Copy(rnnSnapshot.by, by);
        }
    }
}
