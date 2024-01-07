import dark.tensor as dt
import cv2

def save_samples(dataset, model, filename, count = 5):
    image_rows = []

    for i in range(count):
        im, true_mask = dataset[i]
                
        im, true_mask = dt.asarray(im), dt.asarray(true_mask)
        im = dt.expand_dims(im, 0)

        pred_mask = model(im).data
        den = dt.add(1, dt.exp(dt.negative(pred_mask)))
        pred_mask = dt.divide(1, den)

        im        = (im.squeeze()        * 255).astype(dt.uint8)
        pred_mask = (pred_mask.squeeze() * 255).astype(dt.uint8)
        true_mask = (true_mask.squeeze() * 255).astype(dt.uint8)

        row = dt.concatenate([im, dt.stack([true_mask] * 3, axis=0), dt.stack([pred_mask] * 3, axis=0)], axis=1)
        row = dt.rollaxis(row, 0, 3)
        image_rows.append(row)

    image_table = dt.concatenate(image_rows, axis=1)
    cv2.imwrite(filename, dt.numpy(image_table))