using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Training;
using TSFNet.Training.Parameters;

namespace TSFNet.Models.GRU
{
    public class GRU : ITrainable<double[][], GRUBuffers, GRUSnapshot>
    {
        // слои нейронов
        private double[] _input;
        private double[] _hidden;
        private double[] _output;

        private double[] _z;    // вентиль обновления
        private double[] _r;    // вентиль сброса
        private double[] _h;    // кандидат состояния

        // веса
        private double[][] Wz;
        private double[][] Uz;
        private double[] bz;

        private double[][] Wr;
        private double[][] Ur;
        private double[] br;

        private double[][] Wh;
        private double[][] Uh;
        private double[] bh;

        private double[][] V;
        private double[] by;

        public GRU(int inputSize, int hiddenSize, int outputSize)
        {
            // инициализация массивов
            _input = new double[inputSize];
            _hidden = new double[hiddenSize];
            _output = new double[outputSize];

            _z = new double[hiddenSize];
            _r = new double[hiddenSize];
            _h = new double[hiddenSize];

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

            // инициализация весов
            WeightInitializer.Xavier(Wz, inputSize, hiddenSize);
            WeightInitializer.Xavier(Wr, inputSize, hiddenSize);
            WeightInitializer.Xavier(Wh, inputSize, hiddenSize);

            WeightInitializer.Xavier(Uz, hiddenSize, hiddenSize);
            WeightInitializer.Xavier(Ur, hiddenSize, hiddenSize);
            WeightInitializer.Xavier(Uh, hiddenSize, hiddenSize);

            WeightInitializer.Xavier(V, hiddenSize, outputSize);
        }

        public double[] Forward(double[] input)
        {
            Array.Copy(input, _input, input.Length);

            double[] temp = new double[_hidden.Length];

            // вентиль обновления
            LinAlgMath.MulMatVec(Wz, _input, _z);
            LinAlgMath.MulMatVec(Uz, _hidden, temp);
            LinAlgMath.AddVec(_z, temp, _z);
            LinAlgMath.AddVec(_z, bz, _z);
            ActivationFunctions.Sigmoid(_z, _z);

            // вентиль сброса
            LinAlgMath.MulMatVec(Wr, _input, _r);
            LinAlgMath.MulMatVec(Ur, _hidden, temp);
            LinAlgMath.AddVec(_r, temp, _r);
            LinAlgMath.AddVec(_r, br, _r);
            ActivationFunctions.Sigmoid(_r, _r);

            // кандидат нового состояния            
            LinAlgMath.MulElemVec(_r, _hidden, temp);
            LinAlgMath.MulMatVec(Uh, temp, _h);
            LinAlgMath.MulMatVec(Wh, _input, temp);
            LinAlgMath.AddVec(_h, temp, _h);
            LinAlgMath.AddVec(_h, bh, _h);
            ActivationFunctions.Tanh(_h, _h);

            // новое скрытое состояние
            Array.Fill(temp, 1);
            LinAlgMath.SubVec(temp, _z, temp);
            LinAlgMath.MulElemVec(temp, _hidden, _hidden);
            LinAlgMath.MulElemVec(_z, _h, temp);
            LinAlgMath.AddVec(_hidden, temp, _hidden);

            // выход
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

        public void Train(Dataset<double[][]> dataset, Hyperparameters options, GRUBuffers gruBuffers)
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
                        Array.Copy(dataset.GetInput(i + b)[t], gruBuffers.inputHistory[b][t], _input.Length);
                        Forward(dataset.GetInput(i + b)[t]);
                        Array.Copy(_z, gruBuffers.zHistory[b][t], _z.Length);
                        Array.Copy(_r, gruBuffers.rHistory[b][t], _r.Length);
                        Array.Copy(_h, gruBuffers.hHistory[b][t], _h.Length);
                        Array.Copy(_hidden, gruBuffers.hiddenHistory[b][t + 1], _hidden.Length);
                    }

                    LossFunctions.MSEDerivative(dataset.GetTarget(i + b), _output, gruBuffers.dOutputs[b]);
                }

                Backward(gruBuffers, size, options.learningRate, options.l2Lambda, options.threshold);
            }
        }

        private void Backward(GRUBuffers gruBuffers, int size, double learningRate, double l2Lambda, double threshold)
        {
            int sequenceLength = gruBuffers.hiddenHistory[0].Length;

            gruBuffers.ClearAccum();

            // проход по примерам батча
            for (int b = 0; b < size; b++)
            {
                // градиент по V и by
                LinAlgMath.MulOuterVec(gruBuffers.dOutputs[b], gruBuffers.hiddenHistory[b][sequenceLength - 1], gruBuffers.dV);
                LinAlgMath.AddMat(gruBuffers.sumDV, gruBuffers.dV, gruBuffers.sumDV);
                LinAlgMath.AddVec(gruBuffers.sumDby, gruBuffers.dOutputs[b], gruBuffers.sumDby);

                // прокидываем градиент на предыдущий слой
                LinAlgMath.MulMatTVec(V, gruBuffers.dOutputs[b], gruBuffers.delta);

                // проход по развёртке
                for (int sl = sequenceLength - 2; sl >= 0; sl--)
                {
                    // вентиль обновления
                    LinAlgMath.SubVec(gruBuffers.hHistory[b][sl], gruBuffers.hiddenHistory[b][sl], gruBuffers.dz);
                    ActivationFunctions.SigmoidDerivative(gruBuffers.zHistory[b][sl], gruBuffers.derivative);
                    LinAlgMath.MulElemVec(gruBuffers.dz, gruBuffers.derivative, gruBuffers.dz);
                    LinAlgMath.MulElemVec(gruBuffers.dz, gruBuffers.delta, gruBuffers.dz);

                    LinAlgMath.MulOuterVec(gruBuffers.dz, gruBuffers.inputHistory[b][sl], gruBuffers.dW);
                    LinAlgMath.AddMat(gruBuffers.sumDWz, gruBuffers.dW, gruBuffers.sumDWz);

                    LinAlgMath.MulOuterVec(gruBuffers.dz, gruBuffers.hiddenHistory[b][sl], gruBuffers.dU);
                    LinAlgMath.AddMat(gruBuffers.sumDUz, gruBuffers.dU, gruBuffers.sumDUz);

                    LinAlgMath.AddVec(gruBuffers.sumDbz, gruBuffers.dz, gruBuffers.sumDbz);

                    // кандидат нового состояния
                    LinAlgMath.MulElemVec(gruBuffers.delta, gruBuffers.zHistory[b][sl], gruBuffers.dh);
                    ActivationFunctions.TanhDerivative(gruBuffers.hHistory[b][sl], gruBuffers.derivative);
                    LinAlgMath.MulElemVec(gruBuffers.dh, gruBuffers.derivative, gruBuffers.dh);

                    LinAlgMath.MulOuterVec(gruBuffers.dh, gruBuffers.inputHistory[b][sl], gruBuffers.dW);
                    LinAlgMath.AddMat(gruBuffers.sumDWh, gruBuffers.dW, gruBuffers.sumDWh);

                    LinAlgMath.MulElemVec(gruBuffers.rHistory[b][sl], gruBuffers.hiddenHistory[b][sl], gruBuffers.qt);
                    LinAlgMath.MulOuterVec(gruBuffers.dh, gruBuffers.qt, gruBuffers.dU);
                    LinAlgMath.AddMat(gruBuffers.sumDUh, gruBuffers.dU, gruBuffers.sumDUh);

                    LinAlgMath.AddVec(gruBuffers.sumDbh, gruBuffers.dh, gruBuffers.sumDbh);

                    // вентиль сброса
                    LinAlgMath.MulMatTVec(Uh, gruBuffers.dh, gruBuffers.qt);
                    LinAlgMath.MulElemVec(gruBuffers.qt, gruBuffers.hiddenHistory[b][sl], gruBuffers.dr);
                    ActivationFunctions.SigmoidDerivative(gruBuffers.rHistory[b][sl], gruBuffers.derivative);
                    LinAlgMath.MulElemVec(gruBuffers.dr, gruBuffers.derivative, gruBuffers.dr);

                    LinAlgMath.MulOuterVec(gruBuffers.dr, gruBuffers.inputHistory[b][sl], gruBuffers.dW);
                    LinAlgMath.AddMat(gruBuffers.sumDWr, gruBuffers.dW, gruBuffers.sumDWr);

                    LinAlgMath.MulOuterVec(gruBuffers.dr, gruBuffers.hiddenHistory[b][sl], gruBuffers.dU);
                    LinAlgMath.AddMat(gruBuffers.sumDUr, gruBuffers.dU, gruBuffers.sumDUr);

                    LinAlgMath.AddVec(gruBuffers.sumDbr, gruBuffers.dr, gruBuffers.sumDbr);

                    // если есть ещё развёртка - прокидываем градиент на предыдущий слой
                    if (sl > 0)
                    {
                        // (1 - z)
                        Array.Fill(gruBuffers.prevDelta, 1);
                        LinAlgMath.SubVec(gruBuffers.prevDelta, gruBuffers.zHistory[b][sl], gruBuffers.prevDelta);
                        LinAlgMath.MulElemVec(gruBuffers.prevDelta, gruBuffers.delta, gruBuffers.prevDelta);

                        LinAlgMath.MulMatTVec(Uz, gruBuffers.dz, gruBuffers.prevGateDelta);
                        LinAlgMath.AddVec(gruBuffers.prevDelta, gruBuffers.prevGateDelta, gruBuffers.prevDelta);

                        LinAlgMath.MulElemVec(gruBuffers.qt, gruBuffers.rHistory[b][sl], gruBuffers.qt);
                        LinAlgMath.AddVec(gruBuffers.prevDelta, gruBuffers.qt, gruBuffers.prevDelta);

                        LinAlgMath.MulMatTVec(Ur, gruBuffers.dr, gruBuffers.prevGateDelta);
                        LinAlgMath.AddVec(gruBuffers.prevDelta, gruBuffers.prevGateDelta, gruBuffers.prevDelta);

                        Array.Copy(gruBuffers.prevDelta, gruBuffers.delta, _hidden.Length);
                    }
                }
            }

            // апдейт весов
            if (l2Lambda > 0)
            {
                LinAlgMath.MulMatNum(Wz, 1.0 - learningRate * l2Lambda, Wz);
                LinAlgMath.MulMatNum(Wr, 1.0 - learningRate * l2Lambda, Wr);
                LinAlgMath.MulMatNum(Wh, 1.0 - learningRate * l2Lambda, Wh);
                LinAlgMath.MulMatNum(Uz, 1.0 - learningRate * l2Lambda, Uz);
                LinAlgMath.MulMatNum(Ur, 1.0 - learningRate * l2Lambda, Ur);
                LinAlgMath.MulMatNum(Uh, 1.0 - learningRate * l2Lambda, Uh);
                LinAlgMath.MulMatNum(V, 1.0 - learningRate * l2Lambda, V);
            }

            gruBuffers.l2n.AddMatrices(gruBuffers.sumDWz, gruBuffers.sumDUz,
                gruBuffers.sumDWr, gruBuffers.sumDUr, gruBuffers.sumDWh, gruBuffers.sumDUh, gruBuffers.sumDV);
            gruBuffers.l2n.AddVectors(gruBuffers.sumDbz, gruBuffers.sumDbr, gruBuffers.sumDbh, gruBuffers.sumDby);
            gruBuffers.scale = Math.Min(1.0, threshold / gruBuffers.l2n.GetNorm());

            LinAlgMath.MulMatNum(gruBuffers.sumDWz, gruBuffers.scale * learningRate / size, gruBuffers.sumDWz);
            LinAlgMath.SubMat(Wz, gruBuffers.sumDWz, Wz);
            LinAlgMath.MulMatNum(gruBuffers.sumDUz, gruBuffers.scale * learningRate / size, gruBuffers.sumDUz);
            LinAlgMath.SubMat(Uz, gruBuffers.sumDUz, Uz);
            LinAlgMath.MulVecNum(gruBuffers.sumDbz, gruBuffers.scale * learningRate / size, gruBuffers.sumDbz);
            LinAlgMath.SubVec(bz, gruBuffers.sumDbz, bz);

            LinAlgMath.MulMatNum(gruBuffers.sumDWr, gruBuffers.scale * learningRate / size, gruBuffers.sumDWr);
            LinAlgMath.SubMat(Wr, gruBuffers.sumDWr, Wr);
            LinAlgMath.MulMatNum(gruBuffers.sumDUr, gruBuffers.scale * learningRate / size, gruBuffers.sumDUr);
            LinAlgMath.SubMat(Ur, gruBuffers.sumDUr, Ur);
            LinAlgMath.MulVecNum(gruBuffers.sumDbr, gruBuffers.scale * learningRate / size, gruBuffers.sumDbr);
            LinAlgMath.SubVec(br, gruBuffers.sumDbr, br);

            LinAlgMath.MulMatNum(gruBuffers.sumDWh, gruBuffers.scale * learningRate / size, gruBuffers.sumDWh);
            LinAlgMath.SubMat(Wh, gruBuffers.sumDWh, Wh);
            LinAlgMath.MulMatNum(gruBuffers.sumDUh, gruBuffers.scale * learningRate / size, gruBuffers.sumDUh);
            LinAlgMath.SubMat(Uh, gruBuffers.sumDUh, Uh);
            LinAlgMath.MulVecNum(gruBuffers.sumDbh, gruBuffers.scale * learningRate / size, gruBuffers.sumDbh);
            LinAlgMath.SubVec(bh, gruBuffers.sumDbh, bh);

            LinAlgMath.MulMatNum(gruBuffers.sumDV, gruBuffers.scale * learningRate / size, gruBuffers.sumDV);
            LinAlgMath.SubMat(V, gruBuffers.sumDV, V);
            LinAlgMath.MulVecNum(gruBuffers.sumDby, gruBuffers.scale * learningRate / size, gruBuffers.sumDby);
            LinAlgMath.SubVec(by, gruBuffers.sumDby, by);
        }

        public GRUBuffers CreateBuffer(Dataset<double[][]> dataset, Hyperparameters options)
            => new GRUBuffers(_input, _hidden, _output, options.batchSize, dataset.GetRawInput[0].Length);

        public GRUSnapshot CreateSnapshotBuffer()
            => new GRUSnapshot(_input.Length, _hidden.Length, _output.Length);

        public void SaveSnapshot(GRUSnapshot gruSnapshot)
        {
            ArrayCopyUtils.Copy(Wz, gruSnapshot.Wz);
            ArrayCopyUtils.Copy(Uz, gruSnapshot.Uz);
            ArrayCopyUtils.Copy(bz, gruSnapshot.bz);

            ArrayCopyUtils.Copy(Wr, gruSnapshot.Wr);
            ArrayCopyUtils.Copy(Ur, gruSnapshot.Ur);
            ArrayCopyUtils.Copy(br, gruSnapshot.br);

            ArrayCopyUtils.Copy(Wh, gruSnapshot.Wh);
            ArrayCopyUtils.Copy(Uh, gruSnapshot.Uh);
            ArrayCopyUtils.Copy(bh, gruSnapshot.bh);

            ArrayCopyUtils.Copy(V, gruSnapshot.V);
            ArrayCopyUtils.Copy(by, gruSnapshot.by);
        }

        public void RestoreSnapshot(GRUSnapshot gruSnapshot)
        {
            ArrayCopyUtils.Copy(gruSnapshot.Wz, Wz);
            ArrayCopyUtils.Copy(gruSnapshot.Uz, Uz);
            ArrayCopyUtils.Copy(gruSnapshot.bz, bz);

            ArrayCopyUtils.Copy(gruSnapshot.Wr, Wr);
            ArrayCopyUtils.Copy(gruSnapshot.Ur, Ur);
            ArrayCopyUtils.Copy(gruSnapshot.br, br);

            ArrayCopyUtils.Copy(gruSnapshot.Wh, Wh);
            ArrayCopyUtils.Copy(gruSnapshot.Uh, Uh);
            ArrayCopyUtils.Copy(gruSnapshot.bh, bh);

            ArrayCopyUtils.Copy(gruSnapshot.V, V);
            ArrayCopyUtils.Copy(gruSnapshot.by, by);
        }
    }
}
