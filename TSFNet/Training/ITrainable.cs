using TSFNet.Data;
using TSFNet.Training.Parameters;

namespace TSFNet.Training
{
    public interface ITrainable<TInput, TBuffer, TSnapshot>
    {
        double[] Forward(TInput input);
        void Train(Dataset<TInput> dataset, Hyperparameters options, TBuffer buffers);
        TBuffer CreateBuffer(Dataset<TInput> dataset, Hyperparameters options);
        TSnapshot CreateSnapshotBuffer();
        void SaveSnapshot(TSnapshot buffer);
        void RestoreSnapshot(TSnapshot buffer);
    }
}
