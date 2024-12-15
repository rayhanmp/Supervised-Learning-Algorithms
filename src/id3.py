# use the function only 2:
# id3_model, id3_threshold = id3_train(train_data.drop("id", axis=1), "attack_cat") # training
# id3_prediction = id3_predict(id3_threshold, id3_model, val_data.drop("attack_cat", axis=1)) # predicting

from pandas.api.types import is_numeric_dtype
import math
import time

def entropi(dataset, target_feature):
  unique_instances = dataset[target_feature].unique()
  entropi = 0

  for instance in unique_instances:
    probability = (dataset[dataset[target_feature] == instance]).shape[0] / dataset[target_feature].shape[0]
    entropi -= probability * math.log2(probability)

  return entropi
  
def gain(dataset, feature, target_feature):
  init_gain = entropi(dataset, target_feature)
  unique_instances = dataset[feature].unique()
  gain_val = 0

  for instance in unique_instances:
    subset = dataset[dataset[feature] == instance]
    portion = subset.shape[0] / dataset.shape[0]
    calc_entrop = entropi(subset, target_feature)
    gain_val += portion * calc_entrop

  return init_gain - gain_val

def get_best_feature(dataset, target_feature):
  features = (dataset.drop(target_feature, axis=1)).columns.tolist()
  best_feature = None
  best_gain = -1

  for feature in features:
    gain_val = gain(dataset, feature, target_feature)
    if gain_val > best_gain:
      best_feature = feature
      best_gain = gain_val
      
  return best_feature

def transform_to_boolean(threshold, dataset, target_feature):
  try:
    features = (dataset.drop(target_feature, axis=1)).columns.tolist()
  except:
    print("target_feature already dropped")
    features = dataset.columns.tolist()
  i = 0
  # print(len(features))
  for feature in features:
    if is_numeric_dtype(dataset[feature]): #jika bukan numeric baru diubah
      dataset[feature] = dataset[feature] > threshold[i]
    i += 1
  return dataset

def get_breakpoint(dataframe, target_feature):
  breakpoint = []
  prev = dataframe[target_feature][0]
  for i in range(dataframe[target_feature].shape[0]):
    now = dataframe[target_feature][i]
    if prev != now:
      breakpoint.append(i)
  return breakpoint # elemen ke i yang harus di cek sebagai potensi threshold

def get_threshold(breakpoint, subset, feature, target_feature):
  #subset nyimpen 2 columns doang, feature yg di cek samaa target feature nya
  threshold = subset[feature][breakpoint[0]]
  minimum_gain_to_prune = 0.2
  prev_gain = -1.0
  tested_subset = subset.copy()
  gain_val = -1.0
  for i in range (int(len(breakpoint)/10000)):
    # print(i)
    index_bp = breakpoint[i]
    if index_bp <= 0:
      avg = subset[feature][index_bp]
    else:
      # print(subset)
      # print(subset, subset[feature][index_bp], type(subset[feature][index_bp]) , index_bp)
      avg = (subset[feature][index_bp] + subset[feature][(index_bp - 1)]) / 2
    tested_subset[feature] = tested_subset[feature] > avg
    gain_val = gain(tested_subset, feature, target_feature)
    if gain_val > prev_gain:
      prev_gain = gain_val
      threshold = avg
    if gain_val >= minimum_gain_to_prune:
      return threshold
  return threshold
  
def train_get_threshold(dataframe, target_feature):
  dataset = dataframe
  breakpoint = get_breakpoint(dataset, target_feature)
  features = (dataset.drop(target_feature, axis=1)).columns.tolist()

  threshold_for_each_column = []
  for feature in features:
    if is_numeric_dtype(dataset[feature]):
      # print(feature, "feature is a numeric. finding the threshold.")
      threshold = get_threshold(breakpoint, dataframe[[feature, target_feature]], feature, target_feature)
      # print("threshold found:", threshold)
      threshold_for_each_column.append(threshold)
    else: #jika bukan string, kalo string mah gas aja gausah diapa2in berarti udh category soalnya
      threshold_for_each_column.append(int(0))

  return threshold_for_each_column

def make_id3_model(node, target_feature):
  if (len(node[target_feature].unique()) <= 1):
    leaf = node[target_feature].mode()[0]
    return leaf
  if  (len(node.columns.tolist()) <= 1):
        return node[target_feature].mode()[0]

  best_feature = get_best_feature(node, target_feature)
  branch = {best_feature : {}}

  for value in node[best_feature].unique():
    subset = (node.drop(best_feature, axis=1))[node[best_feature] == value]
    # print(subset.columns.tolist())
    subbranch = make_id3_model(subset, target_feature)
    branch[best_feature][value] = subbranch

  return branch

def id3_train(dataframe, target_feature):
  if isinstance(target_feature, list) and len(target_feature) >= 1:
    print("only 1 target feature is allowed in this function")
    return
  else:
    if isinstance(target_feature, list):
      try:
        target_feature = str(target_feature[0])
      except:
        print("what is this?")
        return

  if not pd.api.types.is_string_dtype(dataframe[target_feature]):
    print("target feature data as not numeric is not implemented")
    return

  try:
    dataset = dataframe.drop("id", axis=1).copy()
  except:
    dataset = dataframe.copy()

  # dptin threshold untuk tiap feature
  start_time = time.time()
  print("Getting threshold...")
  threshold = train_get_threshold(dataset, target_feature)
  print("Treshold get with time elapse:", (time.time() - start_time))


  # ubah data numerical jadi boolean
  start_time = time.time()
  print()
  print("Transforming dataset (with a new copy)...")
  dataset = transform_to_boolean(threshold, dataset, target_feature)
  print("Transforming dataset done in", (time.time() - start_time))

  # buat model nya (dalam bentuk dictionary)
  start_time = time.time()
  print()
  print("Making the model...")
  model = make_id3_model(dataset, target_feature)
  print("Model making done in", (time.time() - start_time))

  return model, threshold

def use_id3_model(id3_model, row_data):
    if not isinstance(id3_model, dict):  # Base case: leaf node
        print(f"Reached leaf: {id3_model}")
        return id3_model

    for feature, branch in id3_model.items():
        print(f"Checking feature: {feature}")
        if feature in row_data.index:
            value = row_data[feature]
            print(f"Feature '{feature}' value: {value}")
            if value not in branch:
                print(f"Value '{value}' not found in branch: {branch}")
                raise ValueError(f"Unexpected value '{value}' for feature '{feature}'")
            return use_id3_model(branch[value], row_data.drop(feature))
        else:
            print(f"Feature '{feature}' not found in row data: {row_data.index}")
            raise ValueError("Feature not found in row data")

def id3_predict(threshold, id3_model, dataframe):
    if "id" in dataframe.columns:
        dataset = dataframe.drop("id", axis=1).copy()
    else:
        dataset = dataframe.copy()

    dataset = transform_to_boolean(threshold, dataset, "attack_cat")
    predict = []

    for i, row_data in dataset.iterrows():
        try:
            predict.append(use_id3_model(id3_model, row_data))
        except ValueError as e:
            print(f"Error in row {i}: {e}")
            predict.append(None)  # Handle prediction errors gracefully

    return predict
