import fmeval
from fmeval.model_runners.model_runner import ModelRunner # only use this if you don't already have model outputs
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.data_loaders.data_config import DataConfig
import jsonlines
import pandas as pd
from pandas import DataFrame
from typing import Optional
import os
import boto3
comprehend = boto3.client('comprehend')

class CustomEvaluator(EvalAlgorithmInterface):

    def __init__(self, eval_algorithm_config: EvalAlgorithmConfig):
        """Initialize an instance of a subclass of EvalAlgorithmConfig

        :param eval_algorithm_config: An instance of the subclass of EvalAlgorithmConfig specific to the
                                            current evaluation.
        """
    
    @staticmethod
    def comprehend_eval_algo(model_output: str) -> list:
        """This is a dummy evaluation algorithm that uses the Comprehend Toxicity Detection API on the provided model output

        Args:
            model_output (str): What your actual model output is, this is pre-provided in the example we have

        Returns:
            list: Array of the different toxicity outputs from comprehend
        """

        comprehend_response = comprehend.detect_toxic_content(
            TextSegments=[
                {
                    'Text': model_output
                },
            ],
            LanguageCode='en'
        )
        output = comprehend_response['ResultList'][0]['Labels']
        return output

    
    def evaluate(self, model: Optional[ModelRunner] = None, dataset_config: Optional[DataConfig] = None,
                 prompt_template: Optional[str] = None, save: bool = False, num_records: int = 100) -> str:
        """

        Args:
            model (Optional[ModelRunner], optional): JumpStart Model Runner, don't need one if you already have model output which we do in this case
            dataset_config (Optional[DataConfig], optional): Dataset Configuration with dataset location
            prompt_template (Optional[str], optional): Can structure your prompt depending on what your model is expecting

        Raises:
            FileNotFoundError: If local data file is not found
        """

        # check for dataset in local path, can also implement logic to check for S3
        if dataset_config is not None:
            data_config = [(key, value) for key, value in vars(dataset_config).items()]
            data_location = data_config[1][1] #retrieves the dataset path
            if os.path.isfile(data_location):
                print(f"Detected file: {data_location} in local directory")
            else:
                raise FileNotFoundError(f"The file {data_location} is not in current local directory")

        data = []
        with jsonlines.open(data_location, mode='r') as reader:
            for line in reader:
                model_output = line.get("model_output")
                eval_score = CustomEvaluator.comprehend_eval_algo(model_output)
                line["eval_score"] = eval_score
                data.append(line)
        
        # Create a Pandas DataFrame from the output data
        df = pd.DataFrame(data)
        # write to an output data location in same path, customize this as needed
        output_file = 'custom-eval-results.jsonl'
        print(f"Writing output file with evaluation results: {output_file}")
        with jsonlines.open(output_file, mode='w') as writer:
            for item in df.to_dict(orient='records'):
                writer.write(item)
        return output_file
    
    
    def evaluate_sample(self, model_output: str) -> list:
        """For a single sample model output and target output can be provided.

        Args:
            model_output (str): What your model outputs

        Raises:
            ValueError: If model or target output is not provided

        Returns:
            int: Our evaluation algorithm returns this
        """
        if not model_output:
            raise ValueError("Our custom eval algorithm requires a model output.")
        sample_res = CustomEvaluator.comprehend_eval_algo(model_output)
        return sample_res
