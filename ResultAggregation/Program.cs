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
        /// <summary>
        /// Reads a multiple column headerless CSV
        /// </summary>
        /// <param name="fileName">The path to the CSV</param>
        /// <returns>A list of rows</returns>
        static List<List<string>> ReadMultipleColumnsCSV(string fileName)
        {
            List<List<string>> toReturn = new List<List<string>>();

            using (var reader = new StreamReader(fileName))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    toReturn.Add(new List<string> { values[1], values[2] });
                }
            }

            return toReturn;
        }

        /// <summary>
        /// Reades a single comlumn CSV File, must not have a header
        /// </summary>
        /// <param name="fileName">The path to the CSV</param>
        /// <returns>The content of the CSV</returns>
        static List<string> ReadSingleColumnCSV(string fileName)
        {
            List<string> toReturn = new List<string>();

            using (var reader = new StreamReader(fileName))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();

                    toReturn.Add(line);
                }
            }

            return toReturn;
        }

        static void Main(string[] args)
        {
            List<List<string>> predictedBySVM = new List<List<string>>();
            List<List<string>> predictedByLR = new List<List<string>>();
            List<List<string>> predictedByHumans = new List<List<string>>();

            List<string> mabedTopics = new List<string>();
            List<string> oldaTopics = new List<string>();

            List<List<string>> MabedSets = new List<List<string>>();
            List<List<string>> OldaSets = new List<List<string>>();

            Parallel.Invoke(
                () =>
                {
                    predictedBySVM = ReadMultipleColumnsCSV(Path.Combine("Datasets", "SVMPredictions.csv"));
                },
                () =>
                {
                    predictedByLR = ReadMultipleColumnsCSV(Path.Combine("Datasets", "LRPredictions.csv"));
                },
                () =>
                {
                    predictedByHumans = ReadMultipleColumnsCSV(Path.Combine("Datasets", "[SA]Sentiment140_CleanShave.csv"));
                },
                () =>
                {
                    mabedTopics = ReadSingleColumnCSV(Path.Combine("Datasets", "MABED_CLEAN.csv"));
                    MabedSets = new List<List<string>>();

                    foreach (string topic in mabedTopics)
                    {
                        MabedSets.Add(topic.Split().Distinct().ToList());
                    }
                },
                () =>
                {
                    oldaTopics = ReadSingleColumnCSV(Path.Combine("Datasets", "OLDA_CLEAN.csv"));
                    OldaSets = new List<List<string>>();

                    foreach (string topic in oldaTopics)
                    {
                        OldaSets.Add(topic.Split().Distinct().ToList());
                    }
                }
                );

            int mabedLen = mabedTopics.Count;
            int oldaLen = oldaTopics.Count;
            Console.WriteLine("Loaded  Data");

            ConcurrentDictionary<string, int> MabedSVMScores = new ConcurrentDictionary<string, int>();
            ConcurrentDictionary<string, int> OLDASVMScores = new ConcurrentDictionary<string, int>();

            ConcurrentDictionary<string, int> MabedLRScores = new ConcurrentDictionary<string, int>();
            ConcurrentDictionary<string, int> OLDALRScores = new ConcurrentDictionary<string, int>();

            ConcurrentDictionary<string, int> MabedTrueScores = new ConcurrentDictionary<string, int>();
            ConcurrentDictionary<string, int> OLDATrueScores = new ConcurrentDictionary<string, int>();

            ConcurrentDictionary<string, List<string>> MABEDLabels = new ConcurrentDictionary<string, List<string>>();
            ConcurrentDictionary<string, List<string>> OLDALabels = new ConcurrentDictionary<string, List<string>>();

            ThreadPool.SetMinThreads(12, 12);
            Console.WriteLine("Running True Labels");

            Parallel.ForEach(predictedByHumans, (currentRow) =>
            {
                // Text
                List<string> unique = currentRow[0].Split().Distinct().ToList();

                string tag = currentRow[1];

                for (int i = 0; i < mabedLen; i++)
                {
                    if (unique.Intersect(MabedSets[i]).Count() > 1)
                    {
                        string key = mabedTopics[i];

                        if (MabedTrueScores.TryGetValue(key, out int score))
                        {
                            MabedTrueScores.TryUpdate(key, score + ((tag == "1") ? 1 : -1), score);
                        }
                        else
                        {
                            MabedTrueScores.TryAdd(key, (tag == "1") ? 1 : -1);
                        }
                    }
                }

                for (int i = 0; i < oldaLen; i++)
                {
                    if (unique.Intersect(OldaSets[i]).Count() > 1)
                    {
                        string key = oldaTopics[i];

                        if (OLDATrueScores.TryGetValue(key, out int score))
                        {
                            OLDATrueScores.TryUpdate(key, score + ((tag == "1") ? 1 : -1), score);
                        }
                        else
                        {
                            OLDATrueScores.TryAdd(key, (tag == "1") ? 1 : -1);
                        }
                    }
                }
            });
            Parallel.ForEach(MabedTrueScores, (currentValue) =>
            {
                if (MABEDLabels.TryGetValue(currentValue.Key, out List<string> currentM))
                {
                    List<string> newValue = currentM.Select(x => x).ToList();
                    newValue.Add((currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE");

                    MABEDLabels.TryUpdate(currentValue.Key, newValue, currentM);
                }
                else
                {
                    MABEDLabels.TryAdd(currentValue.Key, new List<string> { (currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE" });
                }
            });
            Parallel.ForEach(OLDATrueScores, (currentValue) =>
           {
               if (OLDALabels.TryGetValue(currentValue.Key, out List<string> currentO))
               {
                   List<string> newValue = currentO.Select(x => x).ToList();
                   newValue.Add((currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE");

                   OLDALabels.TryUpdate(currentValue.Key, newValue, currentO);
               }
               else
               {
                   OLDALabels.TryAdd(currentValue.Key, new List<string> { (currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE" });
               }
           });


            Console.WriteLine("Running SVM Labels");

            Parallel.ForEach(predictedBySVM, (currentRow) =>
            {
                // Text
                List<string> unique = currentRow[0].Split().Distinct().ToList();

                string tag = currentRow[1];

                for (int i = 0; i < mabedLen; i++)
                {
                    if (unique.Intersect(MabedSets[i]).Count() > 1)
                    {
                        string key = mabedTopics[i];

                        if (MabedSVMScores.TryGetValue(key, out int score))
                        {
                            MabedSVMScores.TryUpdate(key, score + ((tag == "1") ? 1 : -1), score);
                        }
                        else
                        {
                            MabedSVMScores.TryAdd(key, (tag == "1") ? 1 : -1);
                        }
                    }
                }

                for (int i = 0; i < oldaLen; i++)
                {
                    if (unique.Intersect(OldaSets[i]).Count() > 1)
                    {
                        string key = oldaTopics[i];

                        if (OLDASVMScores.TryGetValue(key, out int score))
                        {
                            OLDASVMScores.TryUpdate(key, score + ((tag == "1") ? 1 : -1), score);
                        }
                        else
                        {
                            OLDASVMScores.TryAdd(key, (tag == "1") ? 1 : -1);
                        }
                    }
                }
            });
            Parallel.ForEach(MabedSVMScores, (currentValue) =>
            {
                if (MABEDLabels.TryGetValue(currentValue.Key, out List<string> currentM))
                {
                    List<string> newValue = currentM.Select(x => x).ToList();
                    newValue.Add((currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE");

                    MABEDLabels.TryUpdate(currentValue.Key, newValue, currentM);
                }
                else
                {
                    MABEDLabels.TryAdd(currentValue.Key, new List<string> { (currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE" });
                }
            });
            Parallel.ForEach(OLDASVMScores, (currentValue) =>
            {
                if (OLDALabels.TryGetValue(currentValue.Key, out List<string> currentO))
                {
                    List<string> newValue = currentO.Select(x => x).ToList();
                    newValue.Add((currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE");

                    OLDALabels.TryUpdate(currentValue.Key, newValue, currentO);
                }
                else
                {
                    OLDALabels.TryAdd(currentValue.Key, new List<string> { (currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE" });
                }
            });

            Console.WriteLine("Running LR Labels");

            Parallel.ForEach(predictedByLR, (currentRow) =>
            {
                // Text
                List<string> unique = currentRow[0].Split().Distinct().ToList();

                string tag = currentRow[1];

                for (int i = 0; i < mabedLen; i++)
                {
                    if (unique.Intersect(MabedSets[i]).Count() > 1)
                    {
                        string key = mabedTopics[i];

                        if (MabedLRScores.TryGetValue(key, out int score))
                        {
                            MabedLRScores.TryUpdate(key, score + ((tag == "1") ? 1 : -1), score);
                        }
                        else
                        {
                            MabedLRScores.TryAdd(key, (tag == "1") ? 1 : -1);
                        }
                    }
                }

                for (int i = 0; i < oldaLen; i++)
                {
                    if (unique.Intersect(OldaSets[i]).Count() > 1)
                    {
                        string key = oldaTopics[i];

                        if (OLDALRScores.TryGetValue(key, out int score))
                        {
                            OLDALRScores.TryUpdate(key, score + ((tag == "1") ? 1 : -1), score);
                        }
                        else
                        {
                            OLDALRScores.TryAdd(key, (tag == "1") ? 1 : -1);
                        }
                    }
                }
            });
            Parallel.ForEach(MabedLRScores, (currentValue) =>
            {
                if (MABEDLabels.TryGetValue(currentValue.Key, out List<string> currentM))
                {
                    List<string> newValue = currentM.Select(x => x).ToList();
                    newValue.Add((currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE");

                    MABEDLabels.TryUpdate(currentValue.Key, newValue, currentM);
                }
                else
                {
                    MABEDLabels.TryAdd(currentValue.Key, new List<string> { (currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE" });
                }
            });
            Parallel.ForEach(OLDALRScores, (currentValue) =>
            {
                if (OLDALabels.TryGetValue(currentValue.Key, out List<string> currentO))
                {
                    List<string> newValue = currentO.Select(x => x).ToList();
                    newValue.Add((currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE");

                    OLDALabels.TryUpdate(currentValue.Key, newValue, currentO);
                }
                else
                {
                    OLDALabels.TryAdd(currentValue.Key, new List<string> { (currentValue.Value > 0) ? "POSITIVE" : "NEGATIVE" });
                }
            });
            Console.WriteLine("Writing results...");

            var csv = new StringBuilder();
            csv.AppendLine("TOPIC,TRUE,SVM,LR");

            foreach ((string topic, List<string> labels) in MABEDLabels)
            {
                csv.AppendLine($"{topic},{string.Join(",", labels)}");
            }
            File.WriteAllText("MABEDResults.csv", csv.ToString());

            csv.Clear();
            csv.AppendLine("TOPIC,TRUE,SVM,LR");

            foreach ((string topic, List<string> labels) in OLDALabels)
            {
                csv.AppendLine($"{topic},{string.Join(",", labels)}");
            }
            File.WriteAllText("OLDAesults.csv", csv.ToString());
        }
    }
}
