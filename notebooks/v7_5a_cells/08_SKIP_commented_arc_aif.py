# %%writefile arc_aif.py
# import numpy as np
# from scipy.ndimage import label, find_objects
# from collections import Counter

# # -------------------------------------------------------------------
# # 1. Fast Object Extraction (Using C-optimized scipy)
# # -------------------------------------------------------------------
# def extract_object_features(grid):
#     """Extracts connected components (objects) from a grid.
#     Returns: (num_objects, avg_size, list_of_color_hists)
#     """
#     if grid is None or grid.size == 0:
#         return 0, 0, []
    
#     binary = (grid > 0).astype(np.int32)
#     labeled, num_features = label(binary)
    
#     if num_features == 0:
#         return 0, 0, []
    
#     slices = find_objects(labeled)
#     sizes = []
#     color_hists = []
    
#     for sl in slices:
#         # Extract the subgrid for this object
#         obj_mask = labeled[sl] > 0
#         obj_grid = grid[sl][obj_mask]
        
#         # Size (number of cells)
#         sizes.append(np.sum(obj_mask))
        
#         # Color histogram (count of each color 1-9)
#         hist = np.bincount(obj_grid, minlength=10)[1:]  # exclude 0
#         color_hists.append(hist)
    
#     avg_size = np.mean(sizes) if sizes else 0
#     return num_features, avg_size, color_hists


# def compute_color_mapping(hists_in, hists_out):
#     """Attempts to find a color shift (e.g., 1->4, 2->5) between input and output objects."""
#     shifts = []
#     for h_in, h_out in zip(hists_in, hists_out):
#         in_colors = np.where(h_in > 0)[0] + 1
#         out_colors = np.where(h_out > 0)[0] + 1
#         if len(in_colors) == len(out_colors) and len(in_colors) > 0:
#             sorted_in = sorted(in_colors)
#             sorted_out = sorted(out_colors)
#             diffs = np.array(sorted_out) - np.array(sorted_in)
#             if np.all(diffs == diffs[0]) and len(diffs) > 0:
#                 shifts.append(diffs[0])
#     if shifts:
#         return Counter(shifts).most_common(1)[0][0]
#     return 0


# # -------------------------------------------------------------------
# # 2. Main AIF Analysis Function
# # -------------------------------------------------------------------
# def run_aif_analysis(train_data, test_input):
#     """Main function. Returns dict with: confidence, grid, expected_count, color_shift, is_solved."""
#     try:
#         train_features = []
#         input_counts = []
#         output_counts = []
#         all_avg_sizes = []
#         color_shifts = []

#         # Step 1: Analyze all 4 training pairs
#         for ex in train_data:
#             inp_grid = np.array(ex["input"])
#             out_grid = np.array(ex["output"])
            
#             num_in, avg_in, h_in = extract_object_features(inp_grid)
#             num_out, avg_out, h_out = extract_object_features(out_grid)
            
#             train_features.append({
#                 "in": {"num": num_in, "avg": avg_in, "hists": h_in},
#                 "out": {"num": num_out, "avg": avg_out, "hists": h_out}
#             })
#             input_counts.append(num_in)
#             output_counts.append(num_out)
#             all_avg_sizes.append(avg_in)
            
#             if h_in and h_out and len(h_in) == len(h_out):
#                 shift = compute_color_mapping(h_in, h_out)
#                 if shift != 0:
#                     color_shifts.append(shift)

#         # Step 2: Check structural consistency across ALL 4 examples
#         counts_match = (np.std(input_counts) == 0 and np.std(output_counts) == 0 and 
#                         input_counts[0] == output_counts[0])
#         avg_size_stable = np.std(all_avg_sizes) < 0.5
#         expected_count = int(np.mean(input_counts)) if input_counts else 0
#         color_shift = Counter(color_shifts).most_common(1)[0][0] if color_shifts else 0

#         predicted_grid = None
#         confidence = 0.0

#         # Step 3: If consistent, try to predict via simple translation/copy
#         if counts_match and avg_size_stable and expected_count > 0:
#             # Use first training example to compute object movement deltas
#             in_slices = find_objects(label((np.array(train_data[0]["input"]) > 0).astype(int))[0])
#             out_slices = find_objects(label((np.array(train_data[0]["output"]) > 0).astype(int))[0])
            
#             if len(in_slices) == len(out_slices) == expected_count:
#                 test_grid = np.array(test_input)
#                 predicted_grid = np.zeros_like(test_grid)
                
#                 test_binary = (test_grid > 0).astype(int)
#                 test_labels, test_num = label(test_binary)
                
#                 if test_num == expected_count:
#                     test_slices = find_objects(test_labels)
#                     matches = 0
                    
#                     for i, sl_in_test in enumerate(test_slices):
#                         if i >= len(out_slices):
#                             break
#                         # Get object pixels from test
#                         obj_mask = test_labels[sl_in_test] > 0
#                         obj_colors = test_grid[sl_in_test][obj_mask]
                        
#                         # Compute centroid shift from training example 0
#                         dy = out_slices[i][0].start - in_slices[i][0].start
#                         dx = out_slices[i][1].start - in_slices[i][1].start
                        
#                         # Apply shift to test object position
#                         h = sl_in_test[0].stop - sl_in_test[0].start
#                         w = sl_in_test[1].stop - sl_in_test[1].start
#                         new_y = sl_in_test[0].start + dy
#                         new_x = sl_in_test[1].start + dx
                        
#                         # Clamp and place into predicted grid
#                         if (new_y >= 0 and new_x >= 0 and 
#                             new_y + h <= predicted_grid.shape[0] and 
#                             new_x + w <= predicted_grid.shape[1]):
#                             predicted_grid[new_y:new_y+h, new_x:new_x+w][obj_mask] = obj_colors
#                             matches += 1
                    
#                     if matches == expected_count:
#                         confidence = 1.0
#                     else:
#                         confidence = 0.3
#                         predicted_grid = None

#         # Step 4: Return result
#         if predicted_grid is not None and np.sum(predicted_grid) > 0:
#             return {
#                 "confidence": confidence,
#                 "grid": predicted_grid.tolist(),
#                 "expected_count": expected_count,
#                 "color_shift": color_shift,
#                 "is_solved": confidence > 0.95
#             }
#         else:
#             return {
#                 "confidence": max(0.0, 0.1 * (counts_match + avg_size_stable) / 2),
#                 "grid": None,
#                 "expected_count": expected_count,
#                 "color_shift": color_shift,
#                 "is_solved": False
#             }
            
#     except Exception as e:
#         print(f"AIF Error: {e}")
#         return {
#             "confidence": 0.0,
#             "grid": None,
#             "expected_count": 0,
#             "color_shift": 0,
#             "is_solved": False
#         }
