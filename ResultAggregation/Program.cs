using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SAoverED
{
    // This is written in C# as it will run way fater than a regular Python script
    class Program
    {
        const string TRUE_LABELS_KEY = "OriginalDS";
        const int NUM_THREADS = 8;
        const int MINIMAL_INTERSECTION_SIZE = 2;
        const string DATA_FOLDER = "Data";

        /// <summary>
        /// Reads a CSV with the first 2 columns being text and label
        /// </summary>
        /// <param name="fileName">The path to the CSV</param>
        /// <returns>Each row with itsPolarity</returns>
        static Dictionary<string, int> ReadMultipleColumnsCSV(string fileName,
            bool dropHeader = true)
        {
            Dictionary<string, int> toReturn = new Dictionary<string, int>();

            using (var reader = new StreamReader(fileName))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    if (dropHeader)
                    {
                        dropHeader = false;
                        continue;
                    }
                    var values = line.Split(',');

                    try
                    {
                        toReturn.Add(values[0], Convert.ToInt32(values[1]) );
                    }
                    catch { };
                }
            }

            return toReturn;
        }

        /// <summary>
        /// Reades a single comlumn CSV File, must not have a header
        /// </summary>
        /// <param name="fileName">The path to the CSV</param>
        /// <returns>The content of the CSV</returns>
        static List<(string, List<string>)> ReadSingleColumnCSV(string fileName,
            bool dropHeader = true)
        {
            List<(string, List<string>)> toReturn =
                new List<(string, List<string>)>();

            using (var reader = new StreamReader(fileName))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();

                    toReturn.Add((line, line.Split().Distinct().ToList()));
                }
            }
            return dropHeader ? toReturn.Skip(1).ToList() : toReturn;
        }

        static (Dictionary<string, List<(string, List<string>)>>,
                Dictionary<string, Dictionary<string, int>>) ReadData(Dictionary<string, string> filesForTopics,
            Dictionary<string, string> filesForLabels)
        {
            Console.WriteLine("Loading Topics...");
            Dictionary<string, List<(string, List<string>)>> topics =
                new Dictionary<string, List<(string, List<string>)>>();
            foreach (var item in filesForTopics)
            {
                string methodName = item.Key;
                string filePath = item.Value;

                topics[methodName] = ReadSingleColumnCSV(filePath);
            }
            Console.WriteLine($"Loaded {topics.Keys.Count} Sets with about {topics.First().Value.Count} each!");

            Console.WriteLine("Loading Predictions...");
            Dictionary<string, Dictionary<string, int>> predictions =
                new Dictionary<string, Dictionary<string, int>>();
            foreach (var item in filesForLabels)
            {
                string methodName = item.Key;
                string filePath = item.Value;

                predictions[methodName] = ReadMultipleColumnsCSV(filePath);
            }
            Console.WriteLine($"Loaded {predictions.Keys.Count} Sets with about {predictions.First().Value.Count} each!");

            Console.WriteLine("Loaded  Data");

            return (topics, predictions);
        }

        static void RunAggregators(Dictionary<string, string> filesForTopics,
            Dictionary<string, string> filesForLabels)
        {
            ThreadPool.SetMinThreads(NUM_THREADS - 1, 1);

            (Dictionary<string, List<(string, List<string>)>> topics,
                Dictionary<string, Dictionary<string, int>> predictions) = ReadData(filesForTopics, filesForLabels);

            Console.WriteLine("Running Initial Setup...");
            Dictionary<string, int> trueLabels = predictions[TRUE_LABELS_KEY];
            Dictionary<string, List<string>> trueSets = trueLabels.Keys
                .ToDictionary(key => key, value => value.Split().Distinct().ToList());

            List<string> topicGenerators = filesForTopics.Keys.ToList();
            List<string> labelGenerators = filesForLabels.Keys.ToList();

            Dictionary<(string, string), ConcurrentDictionary<string, int>> scores =
                new Dictionary<(string, string), ConcurrentDictionary<string, int>>();

            Console.WriteLine("Aggregating...");
            foreach (string topicGen in topicGenerators)
            {
                foreach (string labelGen in labelGenerators)
                {                        
                    ConcurrentDictionary<string, int> score =
                        new ConcurrentDictionary<string, int>();
                    Console.WriteLine($"Running \"{topicGen}\" X \"{labelGen}\"...");

                    foreach ((string topic, List<string> set) x in topics[topicGen])
                    {
                        Parallel.ForEach(predictions[labelGen], (item) =>
                        {
                            try
                            {
                                string topic = item.Key;
                                int label = item.Value;
                                int valueSign = label == 1 ? 1 : -1;
                                List<string> topicSet = trueSets[topic];

                                if (x.set.Intersect(topicSet).Count() >=
                                MINIMAL_INTERSECTION_SIZE)
                                {
                                    if (score.TryGetValue(topic, out int currentScore))
                                    {
                                        score.TryUpdate(topic, currentScore + valueSign, currentScore);
                                    }
                                    else
                                    {
                                        score.TryAdd(topic, valueSign);
                                    }
                                }
                            }
                            catch { }
                        });
                    }

                    scores[(topicGen, labelGen)] = score;
                    Console.WriteLine($"DONE \"{topicGen}\" X \"{labelGen}\"...");

                }
            }

            Console.WriteLine("Writing results...");

            var csv = new StringBuilder();
            csv.AppendLine("TOPIC_GEN,LABEL_GEN,TOPIC,LABEL");

            foreach(var x in scores)
            {
                (string topicGen, string labelGen) = x.Key;

                foreach(var y in x.Value)
                {
                    string topic = y.Key;
                    int label = y.Value > 0 ? 1 : 0;

                    csv.AppendLine($"{topicGen},{labelGen},{topic},{label}");
                }
            }
            File.WriteAllText("Aggregated.csv", csv.ToString());
            Console.WriteLine("Done!");
        }

        static void GetTopicDS(Dictionary<string, string> filesForTopics,
            Dictionary<string, string> filesForLabels)
        {
            ThreadPool.SetMinThreads(NUM_THREADS - 1, 1);

            (Dictionary<string, List<(string, List<string>)>> topics,
                Dictionary<string, Dictionary<string, int>> predictions) = ReadData(filesForTopics, filesForLabels);

            Console.WriteLine("Running Initial Setup...");
            Dictionary<string, int> trueLabels = predictions[TRUE_LABELS_KEY];
            Dictionary<string, List<string>> trueSets = trueLabels.Keys
                .ToDictionary(key => key, value => value.Split().Distinct().ToList());
            List<string> topicGenerators = filesForTopics.Keys.ToList();
            List<string> labelGenerators = filesForLabels.Keys.ToList();
            List<string> otherGenerators = labelGenerators
                .Where(x => x != TRUE_LABELS_KEY)
                .ToList();

            Console.WriteLine("Aggregating...");

            ConcurrentBag<string> csv = new ConcurrentBag<string>();
            Parallel.ForEach(trueLabels, (x) =>
            {
                string topic = x.Key;
                int label = x.Value;
                StringBuilder rowBuilder = new StringBuilder($"{topic},{label},");
                List<string> topicSet = trueSets[topic];

                foreach (string labelGen in otherGenerators)
                {
                    rowBuilder.Append($"{predictions[labelGen][topic]},");
                }

                foreach (string topicGen in topicGenerators)
                {
                    bool belongsTo = false;
                    foreach ((string topic, List<string> set) y in topics[topicGen])
                    {
                        if (y.set.Intersect(topicSet).Count() >=
                    MINIMAL_INTERSECTION_SIZE)
                        {
                            belongsTo = true;
                            break;
                        }
                    }
                    rowBuilder.Append($"{(belongsTo? 1 : 0)},");
                }

                csv.Add(rowBuilder.ToString().TrimEnd(','));
            });

            string predictors = string.Join(",", labelGenerators.Where(x => x != TRUE_LABELS_KEY));
            string labelers = string.Join(",", topicGenerators.Select(x => $"Is{x}"));
            File.WriteAllText("AggregatedDS.csv", $"Text,Label,{predictors},{labelers}{Environment.NewLine}");
            File.AppendAllLines("AggregatedDS.csv", csv);

            Console.WriteLine("Done!");
        }

        static void AggregationConfiguration()
        {
            RunAggregators(filesForTopics: new Dictionary<string, string>{
                {"MABED", Path.Combine(DATA_FOLDER, "MABED.csv") },
                {"OLDA", Path.Combine(DATA_FOLDER, "OLDA.csv") },
                {"PeakyTopics", Path.Combine(DATA_FOLDER, "PeakyResults.csv")}
            }, filesForLabels: new Dictionary<string, string>
            {
                { TRUE_LABELS_KEY, Path.Combine(DATA_FOLDER, "[Clean][FE]S140C3.csv")},
                { "Bert", Path.Combine(DATA_FOLDER, "[bert][Clean][FE]S140C3.csv")},
                { "RoBERTa", Path.Combine(DATA_FOLDER, "[roberta][Clean][FE]S140C3.csv")},
                { "XLMRoBERTa", Path.Combine(DATA_FOLDER, "[xlmroberta][Clean][FE]S140C3.csv")}
            }
            );
        }

        static void TopicConfiguration()
        {
            GetTopicDS(filesForTopics: new Dictionary<string, string>{
                {"MABED", Path.Combine(DATA_FOLDER, "MABED.csv") },
                {"OLDA", Path.Combine(DATA_FOLDER, "OLDA.csv") },
                {"PeakyTopics", Path.Combine(DATA_FOLDER, "PeakyResults.csv")}
            }, filesForLabels: new Dictionary<string, string>
            {
                { TRUE_LABELS_KEY, Path.Combine(DATA_FOLDER, "[Clean][FE]S140C3.csv")},
                { "Bert", Path.Combine(DATA_FOLDER, "[bert][Clean][FE]S140C3.csv")},
                { "RoBERTa", Path.Combine(DATA_FOLDER, "[roberta][Clean][FE]S140C3.csv")},
                { "XLMRoBERTa", Path.Combine(DATA_FOLDER, "[xlmroberta][Clean][FE]S140C3.csv")}
            }
            );
        }

        static void Main(string[] args)
        {
            TopicConfiguration();
        }
    }
}
