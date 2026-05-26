using TSFNet.Training.Responses;

namespace ForTests
{
    public static class ResponsePrinter
    {
        public static void Print(FitResponse fitResponse)
        {
            Console.WriteLine($"Затрачено времени: {fitResponse.timeElapsed:F2}c");
            if (fitResponse.logTrainLoss.Count > 0)
            {
                Console.WriteLine("Ошибка MSE:");
                foreach (var log in fitResponse.logTrainLoss)
                    Console.WriteLine($"Эпоха {log.Key}:\t{log.Value:F6}");
            }
        }
    }
}
