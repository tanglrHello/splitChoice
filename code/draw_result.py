import matplotlib.pyplot as plt
import pandas as pd

# draw a picture for y_precision/n_precision/y_recall/n_recall/total_precision under the same feature set
# x_axis stands for different y data propagation (i.e. y_prop) in training date
def draw_result_for_features(results, y_props):
    plt.figure(figsize=(20, 10), dpi=80)

    y_precision_series = [result.y_precision for result in results]
    n_precision_series = [result.n_precision for result in results]
    y_recall_series = [result.y_recall for result in results]
    n_recall_series = [result.n_recall for result in results]
    total_precision_series = [result.total_precision for result in results]

    if y_props[0] == False:
        print "change index"
        df = pd.DataFrame({"y_precision":y_precision_series[1:] + [y_precision_series[0]],
                           "n_precision":n_precision_series[1:] + [n_precision_series[0]],
                           "y_recall":y_recall_series[1:] + [y_recall_series[0]],
                           "n_recall":n_recall_series[1:] + [n_recall_series[0]],
                           "total_precision":total_precision_series[1:] + [total_precision_series[0]]},
                          index=y_props[1:] + [y_props[0]])
    else:
        print "don't change index"
        df = pd.DataFrame({"y_precision": y_precision_series,
                           "n_precision": n_precision_series,
                           "y_recall": y_recall_series,
                           "n_recall": n_recall_series,
                           "total_precision": total_precision_series},
                          index=y_props)

    print "show"
    df.plot()
    plt.show()
