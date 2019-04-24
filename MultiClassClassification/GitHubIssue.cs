using Microsoft.ML.Data;

namespace MultiClassClassification
{
    class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }

    }
    /// <summary>
    /// class used for prediction after the model has been trained
    /// </summary>
    class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;

    }
}