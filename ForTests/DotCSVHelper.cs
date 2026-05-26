using CsvHelper.Configuration;
using CsvHelper;
using System.Globalization;

namespace ForTests
{
    public static class DotCSVHelper
    {
        public static List<Dot> Read(string csvPath)
        {
            var config = new CsvConfiguration(CultureInfo.GetCultureInfo("ru-RU"))
            {
                Delimiter = ";",
                PrepareHeaderForMatch = args => args.Header.Trim().TrimStart('\uFEFF').ToLowerInvariant()
            };
            List<Dot> records;
            using (var reader = new StreamReader(csvPath))
            using (var csv = new CsvReader(reader, config))
                records = csv.GetRecords<Dot>().ToList();
            return records;
        }        
    }
}
