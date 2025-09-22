import pytest
import json
import os

from typing import Union

CONF_ERROR_MARGIN = 0.1 # +-0.01 difference allowed in confidence scores
BBOX_ERROR_MARGIN = 50  # +-50 pixels difference allowed in bounding box coordinates


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
   

def test_compare_detections_count(results_ultralytics : dict, results_cpp : dict):

   for model_name in results_ultralytics.keys():

      results_ultralytics_model = results_ultralytics[model_name]["results"]
      results_cpp_model = results_cpp[model_name]["results"]

      for i in range(len(results_ultralytics_model)):

         detections_ultralytics = results_ultralytics_model[i].get("inference_results", [])
         detections_cpp = results_cpp_model[i].get("inference_results", [])

         image_path = results_ultralytics_model[i].get("image_path")

         assert len(detections_ultralytics) == len(detections_cpp), f"Number of detections mismatch for model {model_name}, image :  {image_path}: ultralytics: {len(detections_ultralytics)} != cpp: {len(detections_cpp)}"

def test_compare_detections(results_ultralytics : dict, results_cpp : dict):

   for model_name in results_ultralytics.keys():

      results_ultralytics_model = results_ultralytics[model_name]["results"]
      results_cpp_model = results_cpp[model_name]["results"]

      for i in range(len(results_ultralytics_model)):

         detections_ultralytics = results_ultralytics_model[i].get("inference_results", [])
         detections_cpp = results_cpp_model[i].get("inference_results", [])

         image_path = results_ultralytics_model[i].get("image_path")

         for j in range(len(detections_ultralytics)):

            detection_ultralytics = detections_ultralytics[j]

            class_id_ultralytics = detection_ultralytics.get("class_id")
            conf_ultralytics = detection_ultralytics.get("confidence")
            bbox_ultralytics = detection_ultralytics.get("bbox") 

            for k in range(len(detections_cpp)):

               is_class_found = False

               detection_cpp = detections_cpp[k]

               class_id_cpp = detection_cpp.get("class_id")

               if class_id_ultralytics == class_id_cpp:

                  bbox_cpp = detection_cpp.get("bbox")

                  left_diff = abs(bbox_ultralytics["left"] - bbox_cpp["left"])
                  top_diff = abs(bbox_ultralytics["top"] - bbox_cpp["top"])
                  width_diff = abs(bbox_ultralytics["width"] - bbox_cpp["width"])
                  height_diff = abs(bbox_ultralytics["height"] - bbox_cpp["height"])

                  print(f"""
                  Model: {model_name}, Image: {image_path}, Class ID: {class_id_ultralytics}
                  BBox Ultralytics: {bbox_ultralytics}
                  BBox Cpp: {bbox_cpp}
                  Diffs - Left: {left_diff}, Top: {top_diff}, width: {width_diff}, height: {height_diff}
                  """)

                  if (left_diff <= BBOX_ERROR_MARGIN and
                      top_diff <= BBOX_ERROR_MARGIN and
                      width_diff <= BBOX_ERROR_MARGIN and
                      height_diff <= BBOX_ERROR_MARGIN):

                     conf_cpp = detection_cpp.get("confidence")

                     conf_diff = abs(conf_ultralytics - conf_cpp)

                     print(f"""
                     Model: {model_name}, Image: {image_path}, Class ID: {class_id_ultralytics}
                     Ultralytics Conf: {conf_ultralytics}, Cpp Conf: {conf_cpp}
                     Diff: {conf_diff}
                     """)

                     assert conf_diff <= CONF_ERROR_MARGIN, f"Confidence mismatch for model {model_name}, image :  {image_path}, class_id: {class_id_ultralytics}: ultralytics: {conf_ultralytics} != cpp: {conf_cpp}"

                     is_class_found = True
                     break

            
            assert is_class_found, f"Class ID {class_id_ultralytics} not found in cpp results for model {model_name}, image: {image_path}"

            