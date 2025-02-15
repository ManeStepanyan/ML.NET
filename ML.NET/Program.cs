using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
//using Microsoft.ML.Transforms.Text;
using ML.NET;

namespace BinaryClassification
{
    class Program
    { 
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
              return splitDataView;
            
        }
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // featurize the text column into numeric vector for ML algorithm to understand
            var pipeline = mlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                //add trainer (ml algorithm)
            .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;

        }
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

        }
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
           // UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);
            Console.ReadLine();
           

        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);
            // The Predict() function makes a prediction on a single row of data.
            var res = predictionFunction.Predict(new SentimentData() { SentimentText = "This was a very bad steak" });
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            
            Console.WriteLine($"Prediction {res.Prediction}");
          

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

        }
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
        {
            new SentimentData
            {
            SentimentText = "This was a horrible meal"
            },
            new SentimentData
            {
            SentimentText = "I love this spaghetti."
            }
        };
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            // merging
            IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach ((SentimentData sentiment, SentimentPrediction prediction) item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {item.prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
