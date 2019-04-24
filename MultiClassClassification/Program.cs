using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MultiClassClassification
{

    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        /// <summary>
        /// Extracts and transforms the data, returns the processing pipeline
        /// </summary>
        /// <returns></returns>
        public static IEstimator<ITransformer> ProcessData()
        {
            // The transforms' primary purpose is data featurization. Machine learning algorithms understand featurized data,
            //so the next step is to transform our textual data into a format that our ML algorithms recognize. That format is a numeric vector.
            // As we want to predict the Area GitHub label we copy the Area column into Label column
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
            //featurizes the text (Title and Description) columns into a numeric vector for each called TitleFeaturized and DescriptionFeaturized
            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
            //a learning algorithm processes only features from the Features column
            .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
            //to cache the dataview
            .AppendCacheCheckpoint(_mlContext);
            return pipeline;

        }
        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // choose the lerarning algorithm
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // trains the pipeline
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _trainedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_mlContext);
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            var prediction =  _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;

        }
        /// <summary>
        /// You need to evaluate the model with a different dataset for quality assurance and validation
        /// </summary>
        public static void Evaluate()
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            //You want Micro Accuracy to be as close to 1 as possible.
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");
        }
        /// <summary>
        /// Model can be integrated into any of our .NET applications. Save your trained model to a .zip file
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            //it can be reused and consumed in other applications
            using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

        }
        /// <summary>
        /// deploy and predict with loaded model
        /// </summary>
        private static void PredictIssue()
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = _mlContext.Model.Load(stream);
            }
            GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
            _predEngine = loadedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_mlContext);
            var prediction = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }
        //The problem is to understand what area incoming GitHub issues belong to in order to label them correctly for prioritization and scheduling.
        static void Main(string[] args)
        {
            _mlContext = new MLContext();
            IDataView _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath,hasHeader:true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            Evaluate();
          //  SaveModelAsFile(_mlContext, _trainedModel);
            Console.WriteLine("The model is saved to {0}", _modelPath);
            PredictIssue();
            Console.ReadLine();

        }
    }
}
