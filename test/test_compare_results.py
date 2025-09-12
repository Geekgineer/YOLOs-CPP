import pytest
import json
import os

from typing import Union


@pytest.fixture
def results_ultralytics() -> Union[dict, None]:

   results_ultralytics_path = "results/results_ultralytics.json"

   if not os.path.exists(results_ultralytics_path):
      pytest.skip(f"{results_ultralytics_path} does not exist")
      return None
   
   with open(results_ultralytics_path, "r") as f:
      return json.load(f)

@pytest.fixture
def results_cpp() -> Union[dict, None]:
   results_cpp_path = "results/results_cpp.json"

   if not os.path.exists(results_cpp_path):
      pytest.skip(f"{results_cpp_path} does not exist")
      return None
   
   with open(results_cpp_path, "r") as f:
      return json.load(f)

def test_results_not_empty(results_ultralytics : Union[dict, None], results_cpp : Union[dict, None]):
    
   if results_ultralytics is None:
      pytest.skip("results_ultralytics is None")

   if results_cpp is None:
      pytest.skip("results_cpp is None")

   if len(results_ultralytics) == 0:
      pytest.skip("results_ultralytics is empty")

   if len(results_cpp) == 0:
      pytest.skip("results_cpp is empty")
   
def test_compare_models_names(results_ultralytics : dict, results_cpp : dict):
    
   models_names_ultralytics = set(results_ultralytics.keys())
   models_names_cpp = set(results_cpp.keys())

   for model_name in models_names_ultralytics:
      assert model_name in models_names_cpp, f"Model {model_name} is missing in results_cpp"

def test_compare_weights_paths(results_ultralytics : dict, results_cpp : dict):
    
   for model_name in results_ultralytics.keys():

      weights_path_ultralytics = results_ultralytics[model_name].get("weights_path", "")
      weights_path_cpp = results_cpp[model_name].get("weights_path", "")

      assert weights_path_ultralytics == weights_path_cpp, f"Weights path mismatch for model {model_name}: {weights_path_ultralytics} != {weights_path_cpp}"

def test_compare_images_counts(results_ultralytics : dict, results_cpp : dict):
    
   for model_name in results_ultralytics.keys():

      results_ultralytics_model = results_ultralytics[model_name]["results"]
      results_cpp_model = results_cpp[model_name]["results"]

      assert len(results_ultralytics_model) == len(results_cpp_model), f"Number of results mismatch for model {model_name}: {len(results_ultralytics_model)} != {len(results_cpp_model)}"

def test_compare_images_paths(results_ultralytics : dict, results_cpp : dict):
    
   for model_name in results_ultralytics.keys():

      results_ultralytics_model = results_ultralytics[model_name]["results"]
      results_cpp_model = results_cpp[model_name]["results"]

      for i in range(len(results_ultralytics_model)):

         image_path_ultralytics = results_ultralytics_model[i].get("image_path")
         image_path_cpp = results_cpp_model[i].get("image_path")

         assert image_path_ultralytics == image_path_cpp, f"Image path mismatch for model {model_name}, image {i}: {image_path_ultralytics} != {image_path_cpp}"
   
    

def test_compare_detections_counts(results_ultralytics : dict, results_cpp : dict):

   for model_name in results_ultralytics.keys():

      results_ultralytics_model = results_ultralytics[model_name]["results"]
      results_cpp_model = results_cpp[model_name]["results"]

      for i in range(len(results_ultralytics_model)):

         detections_ultralytics = results_ultralytics_model[i].get("inference_results", [])
         detections_cpp = results_cpp_model[i].get("inference_results", [])

         image_path = results_ultralytics_model[i].get("image_path")

         assert len(detections_ultralytics) == len(detections_cpp), f"""
         Number of detections mismatch for model {model_name}, 
         image :  {image_path}: ultralytics: {len(detections_ultralytics)} != cpp: {len(detections_cpp)}
         """