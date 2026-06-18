using ForTests.Experiments;

namespace ForTests
{
    class Programm
    {
        static void Main(string[] args)
        {
            string csvPath;

            // эксперимент 1
            csvPath = "sin.csv";
            int[] hiddenSizes = { 64, 96, 128, 160, 192, 224, 256 };
            Dictionary<int, Dictionary<string, double>> resultComplexityExperiment = ComplexityExperiment.Run(csvPath, 2, hiddenSizes, repeats: 7);
            ComplexityExperiment.ShowResultComplexityExperiment(resultComplexityExperiment);

            Console.WriteLine("\n\n");

            // эксперимент 2
            csvPath = "sin_trend_noise.csv";
            int[] windowSizes = { 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };
            Dictionary<int, Dictionary<string, double>> resultWindowSizeExperiment = WindowSizeExperiment.Run(csvPath, windowSizes, 32, repeats: 5);
            WindowSizeExperiment.ShowResultWindowSizeExperiment(resultWindowSizeExperiment);

            Console.WriteLine("\n\n");

            // эксперимент 3
            csvPath = "moscow_temp.csv";
            ForecastVisualTest.Run(csvPath, windowSize: 16, hiddenSize: 24);
        }
    }
}
