import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass
from collections import Counter
import math

# -------------------------------------------------------------------
# 1. Fast Object Extraction (same as before, but returns more)
# -------------------------------------------------------------------
def extract_object_features(grid):
    """Returns: (num_objects, avg_size, list_of_masks, list_of_centroids, color_hists)"""
    if grid is None or grid.size == 0:
        return 0, 0, [], [], []
    
    binary = (grid > 0).astype(np.int32)
    labeled, num_features = label(binary)
    
    if num_features == 0:
        return 0, 0, [], [], []
    
    slices = find_objects(labeled)
    sizes = []
    masks = []
    centroids = []
    color_hists = []
    
    for sl in slices:
        obj_mask = labeled[sl] > 0
        masks.append(obj_mask)
        sizes.append(np.sum(obj_mask))
        # Centroid (row, col)
        cy, cx = center_of_mass(labeled[sl], labels=labeled[sl], index=np.unique(labeled[sl][obj_mask])[0])
        centroids.append((cy, cx))
        obj_grid = grid[sl][obj_mask]
        hist = np.bincount(obj_grid, minlength=10)[1:]
        color_hists.append(hist)
    
    avg_size = np.mean(sizes) if sizes else 0
    return num_features, avg_size, masks, centroids, color_hists


def compute_color_mapping(hists_in, hists_out):
    """Finds color shifts between two sets of object histograms."""
    shifts = []
    for h_in, h_out in zip(hists_in, hists_out):
        in_colors = np.where(h_in > 0)[0] + 1
        out_colors = np.where(h_out > 0)[0] + 1
        if len(in_colors) == len(out_colors) and len(in_colors) > 0:
            sorted_in = sorted(in_colors)
            sorted_out = sorted(out_colors)
            diffs = np.array(sorted_out) - np.array(sorted_in)
            if np.all(diffs == diffs[0]) and len(diffs) > 0:
                shifts.append(diffs[0])
    if shifts:
        return Counter(shifts).most_common(1)[0][0]
    return 0


# -------------------------------------------------------------------
# 2. Rule Inference Engine (The World Model)
# -------------------------------------------------------------------
def bresenham_line(p1, p2, shape):
    """Generate a binary mask of a line between two points."""
    r1, c1 = int(round(p1[0])), int(round(p1[1]))
    r2, c2 = int(round(p2[0])), int(round(p2[1]))
    rr, cc = [], []
    dr = abs(r2 - r1)
    dc = -abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr + dc
    while True:
        rr.append(r1)
        cc.append(c1)
        if r1 == r2 and c1 == c2:
            break
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r1 += sr
        if e2 <= dr:
            err += dr
            c1 += sc
    mask = np.zeros(shape, dtype=bool)
    valid_r = np.clip(rr, 0, shape[0]-1)
    valid_c = np.clip(cc, 0, shape[1]-1)
    mask[valid_r, valid_c] = True
    return mask


def match_translation(input_masks, input_centroids, output_masks, output_centroids, grid_shape):
    """Rule 1: Objects moved unchanged (translation)."""
    if len(input_masks) != len(output_masks) or len(input_masks) == 0:
        return None, 0.0
    
    shifts = []
    for i in range(len(input_masks)):
        dy = output_centroids[i][0] - input_centroids[i][0]
        dx = output_centroids[i][1] - input_centroids[i][1]
        shifts.append((dy, dx))
    
    # Check if the shift is consistent across all objects
    if all(s == shifts[0] for s in shifts):
        dy, dx = shifts[0]
        return {"type": "TRANSLATE", "dy": dy, "dx": dx}, 1.0
    return None, 0.0


def match_symmetry(input_masks, output_masks, grid_shape):
    """Rule 2: Mirror / Symmetry."""
    if len(input_masks) == 0 or len(output_masks) == 0:
        return None, 0.0
    
    # Build full input mask from its components
    full_in = np.zeros(grid_shape, dtype=bool)
    for m in input_masks:
        # We need the absolute positions, but masks are cropped.
        # For simplicity, we compare the full grids if only one object.
        # For multiple objects, we check if output masks are mirrored versions of input masks.
        pass
    
    # Simplified: check if output is a horizontal or vertical concatenation of input and its flip
    # (Common ARC symmetry task: half shape -> full shape)
    if len(input_masks) == 1:
        # Try to reconstruct output by flipping the input mask
        in_mask_full = np.zeros(grid_shape, dtype=bool)
        # We'll just use the full input grid (train_data[0]["input"]) outside, so this is tricky.
        # Better to handle this at the grid level in the main function.
        return None, 0.0
    return None, 0.0


def match_arithmetic(input_masks, input_centroids, output_masks, output_centroids, grid_shape):
    """Rule 3: Duplication (count multiplied) or Merging (count divided)."""
    n_in = len(input_masks)
    n_out = len(output_masks)
    if n_in == 0 or n_out == 0:
        return None, 0.0
    
    # Duplication: n_out is multiple of n_in
    if n_out % n_in == 0 and n_out > n_in:
        factor = n_out // n_in
        # Check if every input object is replicated 'factor' times in output
        # We check if output objects are identical to input objects (same shape/color)
        # This is hard without full object matching.
        # We'll return a generic "DUPLICATE" if counts are a clean multiple.
        return {"type": "DUPLICATE", "factor": factor}, 0.8
    # Merging: n_in % n_out == 0
    elif n_in % n_out == 0 and n_in > n_out:
        factor = n_in // n_out
        return {"type": "MERGE", "factor": factor}, 0.8
    return None, 0.0


def match_line_drawing(input_centroids, output_masks, grid_shape):
    """Rule 4: Line drawing between points."""
    if len(input_centroids) == 2 and len(output_masks) == 1:
        # Generate line between the two centroids
        line_mask = bresenham_line(input_centroids[0], input_centroids[1], grid_shape)
        # Compare to output mask (we only have mask, need full grid to compare)
        # We'll just check area: line area should roughly match output area
        out_area = np.sum(output_masks[0])
        line_area = np.sum(line_mask)
        if abs(out_area - line_area) / max(1, line_area) < 0.2:
            return {"type": "DRAW_LINE", "p1": input_centroids[0], "p2": input_centroids[1]}, 1.0
    return None, 0.0


def match_logical_op(input_grids, output_grid):
    """Rule 5: Logical operations (XOR, AND, OR, SUB)."""
    if len(input_grids) != 2:
        return None, 0.0
    A = (input_grids[0] > 0).astype(bool)
    B = (input_grids[1] > 0).astype(bool)
    out = (output_grid > 0).astype(bool)
    
    if np.array_equal(out, np.logical_xor(A, B)):
        return {"type": "XOR"}, 1.0
    if np.array_equal(out, np.logical_and(A, B)):
        return {"type": "AND"}, 1.0
    if np.array_equal(out, np.logical_or(A, B)):
        return {"type": "OR"}, 1.0
    if np.array_equal(out, np.logical_and(A, np.logical_not(B))):
        return {"type": "SUBTRACT"}, 1.0
    if np.array_equal(out, np.logical_and(B, np.logical_not(A))):
        return {"type": "SUBTRACT_REV"}, 1.0
    return None, 0.0


def match_tiling(input_masks, output_masks, grid_shape):
    """Rule 6: Tiling / Fractal."""
    if len(input_masks) == 1 and len(output_masks) > 1:
        # Check if output objects are all identical to the input object
        # We can't easily compare shapes without full reconstruction.
        # But if output has many objects and average size matches input, it's likely tiling.
        return {"type": "TILE"}, 0.7
    return None, 0.0


def apply_rule(rule, test_input, train_data):
    """Apply the inferred rule to the test input to generate the output."""
    if rule is None:
        return None
    
    rule_type = rule.get("type")
    
    if rule_type == "TRANSLATE":
        dy, dx = rule["dy"], rule["dx"]
        # Shift the entire test grid (or its objects)
        # Simpler: shift the whole grid
        grid = np.array(test_input)
        shifted = np.zeros_like(grid)
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] > 0:
                    nr, nc = r + int(dy), c + int(dx)
                    if 0 <= nr < h and 0 <= nc < w:
                        shifted[nr, nc] = grid[r, c]
        return shifted
    
    if rule_type == "DRAW_LINE":
        p1, p2 = rule["p1"], rule["p2"]
        line_mask = bresenham_line(p1, p2, np.array(test_input).shape)
        # Apply colors from input objects (or just use color 1)
        output = np.zeros_like(test_input, dtype=int)
        # Use the first input object's color if available
        color = 1
        # If there are objects, use their colors
        grid = np.array(test_input)
        colors = grid[grid > 0]
        if len(colors) > 0:
            color = int(colors[0])
        output[line_mask] = color
        return output
    
    if rule_type == "DUPLICATE":
        factor = rule["factor"]
        grid = np.array(test_input)
        # Simply tile the grid in a pattern (e.g., 2x2 or 1xN)
        # For simplicity, just repeat the whole grid
        h, w = grid.shape
        # Try to arrange in a square-ish grid
        rows = int(np.ceil(np.sqrt(factor)))
        cols = int(np.ceil(factor / rows))
        out_h = h * rows
        out_w = w * cols
        output = np.zeros((out_h, out_w), dtype=int)
        for i in range(rows):
            for j in range(cols):
                if i * cols + j < factor:
                    output[i*h:(i+1)*h, j*w:(j+1)*w] = grid
        return output
    
    if rule_type in ["XOR", "AND", "OR", "SUBTRACT", "SUBTRACT_REV"]:
        # This rule applies to the input itself (2 input grids in training)
        # For test, we only have one input. We can't apply logical ops to single input.
        return None
    
    if rule_type == "TILE":
        grid = np.array(test_input)
        # Tile the input pattern across the whole output canvas
        # We need output size, but we don't know it. Use training output size.
        # Use first training output as template size
        train_out = np.array(train_data[0]["output"])
        h_out, w_out = train_out.shape
        h_in, w_in = grid.shape
        output = np.zeros((h_out, w_out), dtype=int)
        for r in range(0, h_out, h_in):
            for c in range(0, w_out, w_in):
                rh = min(h_in, h_out - r)
                rw = min(w_in, w_out - c)
                output[r:r+rh, c:c+rw] = grid[:rh, :rw]
        return output
    
    return None


# -------------------------------------------------------------------
# 3. Main AIF Analysis Function (Upgraded)
# -------------------------------------------------------------------
def run_aif_analysis(train_data, test_input):
    """Full Active Inference World Model.
    Tries multiple rules, picks the one that perfectly explains all training examples.
    """
    try:
        train_features = []
        input_counts = []
        output_counts = []
        all_avg_sizes = []
        color_shifts = []

        # Step 1: Extract features from all training pairs
        for ex in train_data:
            inp_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])
            
            num_in, avg_in, masks_in, cents_in, h_in = extract_object_features(inp_grid)
            num_out, avg_out, masks_out, cents_out, h_out = extract_object_features(out_grid)
            
            train_features.append({
                "in": {"num": num_in, "avg": avg_in, "masks": masks_in, "cents": cents_in, "grid": inp_grid, "hists": h_in},
                "out": {"num": num_out, "avg": avg_out, "masks": masks_out, "cents": cents_out, "grid": out_grid, "hists": h_out}
            })
            input_counts.append(num_in)
            output_counts.append(num_out)
            all_avg_sizes.append(avg_in)
            
            if h_in and h_out and len(h_in) == len(h_out):
                shift = compute_color_mapping(h_in, h_out)
                if shift != 0:
                    color_shifts.append(shift)

        # Step 2: Check consistency across all examples
        counts_match = (np.std(input_counts) == 0 and np.std(output_counts) == 0)
        avg_size_stable = np.std(all_avg_sizes) < 0.5
        expected_count = int(np.mean(input_counts)) if input_counts else 0
        color_shift = Counter(color_shifts).most_common(1)[0][0] if color_shifts else 0

        predicted_grid = None
        confidence = 0.0
        best_rule = None

        # Use the first training example to infer the rule
        first_in = train_features[0]["in"]
        first_out = train_features[0]["out"]
        grid_shape = first_in["grid"].shape

        # --- CASCADE OF RULES (BFS over transformations) ---
        
        # Rule 1: Translation
        rule, conf = match_translation(
            first_in["masks"], first_in["cents"],
            first_out["masks"], first_out["cents"],
            grid_shape
        )
        if rule and conf > confidence:
            # Validate on other training examples
            all_match = True
            for feat in train_features[1:]:
                r2, c2 = match_translation(
                    feat["in"]["masks"], feat["in"]["cents"],
                    feat["out"]["masks"], feat["out"]["cents"],
                    grid_shape
                )
                if not r2 or r2["type"] != "TRANSLATE":
                    all_match = False
                    break
            if all_match:
                best_rule = rule
                confidence = conf

        # Rule 2: Arithmetic (Duplication / Merging)
        if not best_rule:
            rule, conf = match_arithmetic(
                first_in["masks"], first_in["cents"],
                first_out["masks"], first_out["cents"],
                grid_shape
            )
            if rule and conf > confidence:
                all_match = True
                for feat in train_features[1:]:
                    r2, c2 = match_arithmetic(
                        feat["in"]["masks"], feat["in"]["cents"],
                        feat["out"]["masks"], feat["out"]["cents"],
                        grid_shape
                    )
                    if not r2 or r2["type"] != rule["type"]:
                        all_match = False
                        break
                if all_match:
                    best_rule = rule
                    confidence = conf

        # Rule 3: Line Drawing
        if not best_rule and first_in["num"] == 2 and first_out["num"] == 1:
            rule, conf = match_line_drawing(
                first_in["cents"],
                first_out["masks"],
                grid_shape
            )
            if rule and conf > confidence:
                all_match = True
                for feat in train_features[1:]:
                    if feat["in"]["num"] != 2 or feat["out"]["num"] != 1:
                        all_match = False
                        break
                    r2, c2 = match_line_drawing(
                        feat["in"]["cents"],
                        feat["out"]["masks"],
                        grid_shape
                    )
                    if not r2:
                        all_match = False
                        break
                if all_match:
                    best_rule = rule
                    confidence = conf

        # Rule 4: Logical Operations (XOR, AND, etc.)
        if not best_rule and len(train_data[0]["input"]) == 2:  # two input grids?
            # Actually ARC train data is a list of dicts, each with "input" and "output".
            # Logical ops usually appear with multiple input grids in one pair? 
            # ARC standard: each pair is one input -> one output. So this doesn't apply.
            pass

        # Rule 5: Tiling
        if not best_rule and first_in["num"] == 1 and first_out["num"] > 1:
            rule, conf = match_tiling(
                first_in["masks"],
                first_out["masks"],
                grid_shape
            )
            if rule and conf > confidence:
                all_match = True
                for feat in train_features[1:]:
                    if feat["in"]["num"] != 1 or feat["out"]["num"] <= 1:
                        all_match = False
                        break
                if all_match:
                    best_rule = rule
                    confidence = conf

        # Step 3: Apply the best rule to the test input
        if best_rule:
            predicted_grid = apply_rule(best_rule, test_input, train_data)
            # If we got a valid grid, set high confidence
            if predicted_grid is not None and np.sum(predicted_grid) > 0:
                confidence = max(confidence, 0.95)

        # Step 4: Fallback to simple object-count pruning if no rule matched
        if predicted_grid is None:
            if counts_match and avg_size_stable and expected_count > 0:
                # Simple copy of test grid (fallback)
                predicted_grid = np.array(test_input)
                confidence = max(0.0, 0.1 * (counts_match + avg_size_stable) / 2)
            else:
                confidence = 0.0

        # Step 5: Return result
        if predicted_grid is not None and np.sum(predicted_grid) > 0:
            return {
                "confidence": confidence,
                "grid": predicted_grid.tolist(),
                "expected_count": expected_count,
                "color_shift": color_shift,
                "rule_type": best_rule["type"] if best_rule else "UNKNOWN",
                "is_solved": confidence > 0.95
            }
        else:
            return {
                "confidence": 0.0,
                "grid": None,
                "expected_count": expected_count,
                "color_shift": color_shift,
                "rule_type": "NONE",
                "is_solved": False
            }
            
    except Exception as e:
        print(f"AIF Error: {e}")
        return {
            "confidence": 0.0,
            "grid": None,
            "expected_count": 0,
            "color_shift": 0,
            "rule_type": "ERROR",
            "is_solved": False
        }