import numpy as np
from bounding_box import iou, convert_coordinates

def greedy_nms(pred_decoded, iou_threshold=0.45, coords='corners', border_pixels='half'):
    pred_decoded_nms = []
    for batch_item in pred_decoded: # For the labels of each batch item...
        boxes_left = np.copy(batch_item)
        maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
            maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
            maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
            maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
            similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        pred_decoded_nms.append(np.array(maxima))

    return pred_decoded_nms
def _greedy_nms(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_detections()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)
def _greedy_nms2(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function in `decode_detections_fast()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)
def decode_detections(pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_h=None,
                      img_w=None,
                      border_pixels='half'):
    if normalize_coords and ((img_h is None) or (img_w is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_h == {}` and `img_w == {}`".format(img_h, img_w))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    pred_decoded_raw = np.copy(pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        pred_decoded_raw[:,:,[-2,-1]] = np.exp(pred_decoded_raw[:,:,[-2,-1]] * pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        pred_decoded_raw[:,:,[-2,-1]] *= pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        pred_decoded_raw[:,:,[-4,-3]] *= pred[:,:,[-4,-3]] * pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        pred_decoded_raw[:,:,[-4,-3]] += pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        pred_decoded_raw = convert_coordinates(pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        pred_decoded_raw[:,:,-4:] *= pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(pred[:,:,-7] - pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(pred[:,:,-5] - pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        pred_decoded_raw[:,:,-4:] += pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        pred_decoded_raw = convert_coordinates(pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        pred_decoded_raw[:,:,-4:] *= pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(pred[:,:,-6] - pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(pred[:,:,-5] - pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        pred_decoded_raw[:,:,-4:] += pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        pred_decoded_raw[:,:,[-4,-2]] *= img_w # Convert xmin, xmax back to absolute coordinates
        pred_decoded_raw[:,:,[-3,-1]] *= img_h # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    pred_decoded = [] # Store the final predictions in this list
    for batch_item in pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return pred_decoded
def decode_detections_fast(pred,
                           confidence_thresh=0.5,
                           iou_threshold=0.45,
                           top_k='all',
                           input_coords='centroids',
                           normalize_coords=True,
                           img_h=None,
                           img_w=None,
                           border_pixels='half'):
    if normalize_coords and ((img_h is None) or (img_w is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_h == {}` and `img_w == {}`".format(img_h, img_w))

    # 1: Convert the classes from one-hot encoding to their class ID
    pred_converted = np.copy(pred[:,:,-14:-8]) # Slice out the four offset predictions plus two elements whereto we'll write the class IDs and confidences in the next step
    pred_converted[:,:,0] = np.argmax(pred[:,:,:-12], axis=-1) # The indices of the highest confidence values in the one-hot class vectors are the class ID
    pred_converted[:,:,1] = np.amax(pred[:,:,:-12], axis=-1) # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        pred_converted[:,:,[4,5]] = np.exp(pred_converted[:,:,[4,5]] * pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        pred_converted[:,:,[4,5]] *= pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        pred_converted[:,:,[2,3]] *= pred[:,:,[-4,-3]] * pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        pred_converted[:,:,[2,3]] += pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        pred_converted = convert_coordinates(pred_converted, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        pred_converted[:,:,2:] *= pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        pred_converted[:,:,[2,3]] *= np.expand_dims(pred[:,:,-7] - pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        pred_converted[:,:,[4,5]] *= np.expand_dims(pred[:,:,-5] - pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        pred_converted[:,:,2:] += pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        pred_converted = convert_coordinates(pred_converted, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        pred_converted[:,:,2:] *= pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        pred_converted[:,:,[2,4]] *= np.expand_dims(pred[:,:,-6] - pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        pred_converted[:,:,[3,5]] *= np.expand_dims(pred[:,:,-5] - pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        pred_converted[:,:,2:] += pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        pred_converted[:,:,[2,4]] *= img_w # Convert xmin, xmax back to absolute coordinates
        pred_converted[:,:,[3,5]] *= img_h # Convert ymin, ymax back to absolute coordinates

    # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
    pred_decoded = []
    for batch_item in pred_converted: # For each image in the batch...
        boxes = batch_item[np.nonzero(batch_item[:,0])] # ...get all boxes that don't belong to the background class,...
        boxes = boxes[boxes[:,1] >= confidence_thresh] # ...then filter out those positive boxes for which the prediction confidence is too low and after that...
        if iou_threshold: # ...if an IoU threshold is set...
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on the remaining boxes.
        if top_k != 'all' and boxes.shape[0] > top_k: # If we have more than `top_k` results left at this point...
            top_k_indices = np.argpartition(boxes[:,1], kth=boxes.shape[0]-top_k, axis=0)[boxes.shape[0]-top_k:] # ...get the indices of the `top_k` highest-scoring boxes...
            boxes = boxes[top_k_indices] # ...and keep only those boxes...
        pred_decoded.append(boxes) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return pred_decoded
def decode_detections_debug(pred,
                            confidence_thresh=0.01,
                            iou_threshold=0.45,
                            top_k=200,
                            input_coords='centroids',
                            normalize_coords=True,
                            img_h=None,
                            img_w=None,
                            variance_encoded_in_target=False,
                            border_pixels='half'):
     if normalize_coords and ((img_h is None) or (img_w is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_h == {}` and `img_w == {}`".format(img_h, img_w))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

     pred_decoded_raw = np.copy(pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

     if input_coords == 'centroids':
        if variance_encoded_in_target:
            # Decode the predicted box center x and y coordinates.
            pred_decoded_raw[:,:,[-4,-3]] = pred_decoded_raw[:,:,[-4,-3]] * pred[:,:,[-6,-5]] + pred[:,:,[-8,-7]]
            # Decode the predicted box width and heigt.
            pred_decoded_raw[:,:,[-2,-1]] = np.exp(pred_decoded_raw[:,:,[-2,-1]]) * pred[:,:,[-6,-5]]
        else:
            # Decode the predicted box center x and y coordinates.
            pred_decoded_raw[:,:,[-4,-3]] = pred_decoded_raw[:,:,[-4,-3]] * pred[:,:,[-6,-5]] * pred[:,:,[-4,-3]] + pred[:,:,[-8,-7]]
            # Decode the predicted box width and heigt.
            pred_decoded_raw[:,:,[-2,-1]] = np.exp(pred_decoded_raw[:,:,[-2,-1]] * pred[:,:,[-2,-1]]) * pred[:,:,[-6,-5]]
        pred_decoded_raw = convert_coordinates(pred_decoded_raw, start_index=-4, conversion='centroids2corners')
     elif input_coords == 'minmax':
        pred_decoded_raw[:,:,-4:] *= pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(pred[:,:,-7] - pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(pred[:,:,-5] - pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        pred_decoded_raw[:,:,-4:] += pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        pred_decoded_raw = convert_coordinates(pred_decoded_raw, start_index=-4, conversion='minmax2corners')
     elif input_coords == 'corners':
        pred_decoded_raw[:,:,-4:] *= pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(pred[:,:,-6] - pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(pred[:,:,-5] - pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        pred_decoded_raw[:,:,-4:] += pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
     else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

     if normalize_coords:
        pred_decoded_raw[:,:,[-4,-2]] *= img_w # Convert xmin, xmax back to absolute coordinates
        pred_decoded_raw[:,:,[-3,-1]] *= img_h # Convert ymin, ymax back to absolute coordinates

    # 3: For each batch item, prepend each box's internal index to its coordinates.

     pred_decoded_raw2 = np.zeros((pred_decoded_raw.shape[0], pred_decoded_raw.shape[1], pred_decoded_raw.shape[2] + 1)) # Expand the last axis by one.
     pred_decoded_raw2[:,:,1:] = pred_decoded_raw
     pred_decoded_raw2[:,:,0] = np.arange(pred_decoded_raw.shape[1]) # Put the box indices as the first element for each box via broadcasting.
     pred_decoded_raw = pred_decoded_raw2

    # 4: Apply confidence thresholding and non-maximum suppression per class

     n_classes = pred_decoded_raw.shape[-1] - 5 # The number of classes is the length of the last axis minus the four box coordinates and minus the index

     pred_decoded = [] # Store the final predictions in this list
     for batch_item in pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
             single_class = batch_item[:,[0, class_id + 1, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 6]` and...
             threshold_met = single_class[single_class[:,1] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
             if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                 maxima = _greedy_nms_debug(threshold_met, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on them.
                 maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                 maxima_output[:,0] = maxima[:,0] # Write the box index to the first column...
                 maxima_output[:,1] = class_id # ...and write the class ID to the second column...
                 maxima_output[:,2:] = maxima[:,1:] # ...and write the rest of the maxima data to the other columns...
                 pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        pred = np.concatenate(pred, axis=0)
        if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
            top_k_indices = np.argpartition(pred[:,2], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
            pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

     return pred_decoded
def _greedy_nms_debug(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_detections_debug()`. The difference is that it keeps the indices of all
    left-over boxes for each batch item, which allows you to know which predictor layer predicted a given output
    box and is thus useful for debugging.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)
def get_num_boxes_per_pred_layer(predictor_sizes, aspect_ratios, two_boxes_for_ar1):
    '''
    Returns a list of the number of boxes that each predictor layer predicts.
    `aspect_ratios` must be a nested list, containing a list of aspect ratios
    for each predictor layer.
    '''
    num_boxes_per_pred_layer = []
    for i in range(len(predictor_sizes)):
        if two_boxes_for_ar1:
            num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * (len(aspect_ratios[i]) + 1))
        else:
            num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * len(aspect_ratios[i]))
    return num_boxes_per_pred_layer
def get_pred_layers(pred_decoded, num_boxes_per_pred_layer):
    '''
    For a given prediction tensor decoded with `decode_detections_debug()`, returns a list
    with the indices of the predictor layers that made each predictions.
    That is, this function lets you know which predictor layer is responsible
    for a given prediction.ss
    Arguments:
        pred_decoded (array): The decoded model output tensor. Must have been
            decoded with `decode_detections_debug()` so that it contains the internal box index
            for each predicted box.
        num_boxes_per_pred_layer (list): A list that contains the total number
            of boxes that each predictor layer predicts.
    '''
    pred_layers_all = []
    cum_boxes_per_pred_layer = np.cumsum(num_boxes_per_pred_layer)
    for batch_item in pred_decoded:
        pred_layers = []
        for prediction in batch_item:
            if (prediction[0] < 0) or (prediction[0] >= cum_boxes_per_pred_layer[-1]):
                raise ValueError("Box index is out of bounds of the possible indices as given by the values in `num_boxes_per_pred_layer`.")
            for i in range(len(cum_boxes_per_pred_layer)):
                if prediction[0] < cum_boxes_per_pred_layer[i]:
                    pred_layers.append(i)
                    break
        pred_layers_all.append(pred_layers)
    return pred_layers_all