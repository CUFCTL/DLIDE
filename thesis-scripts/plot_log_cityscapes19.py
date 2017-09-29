import inspect, os, random, time, sys
import matplotlib.cm as cmx 
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks

################################################################################
################################################################################

def get_log_parsing_script():  
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return dirname + '/parse_log_cityscapes19.sh'

def print_help():
    print 'Usage: '
    print '    ./plot_log.py <log_file> <chart_type>'
    print 'Supported chart types:'
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    for i in xrange(num):
        print '    %d: %s' % (i, supported_chart_types[i])
    sys.exit()

################################################################################

def create_field_index():
    train_key = 'Train'
    test_key = 'Test'
    field_index = {train_key:{'Iters':0, train_key + ' LearningRate':1, train_key + ' Accuracy':2, train_key + ' Loss':3, train_key + ' Road':4, train_key + ' Sidewalk':5, 
                                train_key + ' Building':6, train_key + ' Wall':7, train_key + ' Fence':8, train_key + ' Pole':9, train_key + ' TrafficLight':10,
                                train_key + ' TrafficSign':11, train_key + ' Vegetation':12, train_key + ' Terrain':13, train_key + ' Sky':14, train_key + ' Person':15, 
                                train_key + ' Rider':16, train_key + ' Car':17, train_key + ' Truck':18, train_key + ' Bus':19, train_key + ' Train':20, 
                                train_key + ' Motorcycle':21, train_key + ' Bicycle': 22},
                   test_key:{'Iters':0, test_key + ' Accuracy':1, test_key + ' Loss':2, test_key + ' Road':3, test_key + ' Sidewalk':4, train_key + ' Building':5, 
                                test_key + ' Wall':6, test_key + ' Fence':7, test_key + ' Pole':8, test_key + ' TrafficLight':9, test_key + ' TrafficSign':10, 
                                test_key + ' Vegetation':11, test_key + ' Terrain':12, test_key + ' Sky':13, test_key + ' Person':14, test_key + ' Rider':15, 
                                test_key + ' Car':16, test_key + ' Truck':17, test_key + ' Bus':18, test_key + ' Train':19, test_key + ' Motorcycle':20, 
                                test_key + ' Bicycle': 21}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def is_x_axis_field(field):
    x_axis_fields = ['Iters']
    return field in x_axis_fields

def get_axis_fields():
    x_axis_field_list = []
    y_axis_field_list = []
    field_index, field = create_field_index()
    for f in xrange(len(field)):
        if is_x_axis_field(field[f]):
            x_axis_field_list.append(field[f])
        else:
            y_axis_field_list.append(field[f])
    return x_axis_field_list, y_axis_field_list

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(' vs ')
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field

def get_field_indices(x_axis_field, y_axis_field, chart_type):
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index()[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]

################################################################################

def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in xrange(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in xrange(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (fields[i], ' vs ', fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    return get_supported_chart_types()[chart_type]

################################################################################

def get_data_file(chart_type, log_file):
    return (log_file + '.' + get_data_file_type(chart_type).lower())

def get_data_file_type(chart_type):
    return get_chart_type_description(chart_type).split()[0]

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] != '#':
                fields = line.split()
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data

################################################################################

def plot_train(log_file):
    linewidth = 1.5
    train_fields = []
    x_axis_fields, y_axis_fields = get_axis_fields()
    for y in y_axis_fields:
        if y.split()[0] == 'Train' and y.split()[1] != 'Accuracy' and y.split()[1] != 'Loss':
            train_fields.append(y)

    # plot test plots
    chart_type = 19 # hardcoded
    for train in train_fields:
        print train
        x, y = get_field_indices('Iters', train, chart_type)
        data = load_data(log_file + '.train', x, y)
        color = [random.random(), random.random(), random.random()]
        plt.plot(data[0], data[1], label=str(train.split()[1]), color=color, marker='x', linewidth=linewidth)
    plt.ylim(ymax=1, ymin=0)
    plt.xlabel('Iters')
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    plt.title('Train accuracy')
    # plt.savefig('train_' + path_to_png, bbox_inches='tight')
    plt.show()

def plot_test(log_file):
    linewidth = 1.5
    test_fields = []
    x_axis_fields, y_axis_fields = get_axis_fields()
    for y in y_axis_fields:
        if y.split()[0] == 'Test' and y.split()[1] != 'Accuracy' and y.split()[1] != 'Loss':
            test_fields.append(y)

    # plot test plots
    chart_type = 1 # hardcoded
    for test in test_fields:
        print test
        x, y = get_field_indices('Iters', test, chart_type)
        data = load_data(log_file + '.test', x, y)
        color = [random.random(), random.random(), random.random()]
        plt.plot(data[0], data[1], label=str(test.split()[1]), color=color, marker='x', linewidth=linewidth)
    plt.ylim(ymax=1, ymin=0)
    plt.xlabel('Iters')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Test accuracy')
    # plt.savefig('test_' + path_to_png, bbox_inches='tight')
    plt.show()

def plot_chart(chart_type, log_file):
    data_file = get_data_file(chart_type, log_file)
    x_axis_field, y_axis_field = get_field_descriptions(chart_type)
    x, y = get_field_indices(x_axis_field, y_axis_field, chart_type)
    data = load_data(data_file, x, y)
    label = log_file[log_file.rfind('/')+1 : log_file.rfind('.log')]

    color = [random.random(), random.random(), random.random()]
    linewidth = 1.5
    plt.plot(data[0], data[1], label=label, color=color, marker='x', linewidth=linewidth)
    
    plt.xlabel(x_axis_field)
    if chart_type != 26:
        plt.ylim(ymax=max(data[1])*1.2, ymin=0)
    plt.legend(loc='upper right')
    plt.title(get_chart_type_description(chart_type))
    plt.show()

def plot_chart_save(chart_type, log_file, out_dir):
    data_file = get_data_file(chart_type, log_file)
    x_axis_field, y_axis_field = get_field_descriptions(chart_type)
    x, y = get_field_indices(x_axis_field, y_axis_field, chart_type)
    data = load_data(data_file, x, y)
    label = log_file[log_file.rfind('/')+1 : log_file.rfind('.log')]

    color = [random.random(), random.random(), random.random()]
    linewidth = 1.5
    plt.plot(data[0], data[1], label=label, color=color, marker='x', linewidth=linewidth)
    
    plt.xlabel(x_axis_field)
    if chart_type == 26:
        plt.ylim(ymax=max(data[1])*1.2, ymin=0)
    elif chart_type == 5 or chart_type == 27:
        plt.ylim(ymax=10, ymin=0)
    else:
        plt.ylim(ymax=1, ymin=0)
    # plt.legend(loc='upper right')
    # plt.title(get_chart_type_description(chart_type))
    plt.savefig(out_dir + get_chart_type_description(chart_type) + '.png', bbox_inches='tight')
    plt.clf()


def plot_model(chart_type, log_file):
    data_file = get_data_file(20, log_file)
    x_axis_field, y_axis_field = get_field_descriptions(20)
    x, y = get_field_indices(x_axis_field, y_axis_field, 20)
    data = load_data(data_file, x, y)

    color = [random.random(), random.random(), random.random()]
    linewidth = 1.5
    plt.plot(data[0], data[1], label='Train accuracy', color=color, marker='x', linewidth=linewidth)

    ###################################

    data_file = get_data_file(5, log_file)
    x_axis_field, y_axis_field = get_field_descriptions(5)

    x, y = get_field_indices(x_axis_field, y_axis_field, 5)
    data = load_data(data_file, x, y)

    color = [random.random(), random.random(), random.random()]
    linewidth = 1.5
    plt.plot(data[0], data[1], label='Test loss', color=color, marker='x', linewidth=linewidth)
    
    plt.xlabel(x_axis_field)
    plt.legend(loc='upper right')
    plt.title('test')
    plt.ylim(ymax=1, ymin=0)

    plt.show()

################################################################################
################################################################################

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_help()
    else:
        if not os.path.exists(sys.argv[1]):
            print 'error: log file does not exist: %s\n' % sys.argv[1]
            print_help()
            sys.exit()
        
        # parse log and plot
        chart_type = int(sys.argv[2])

        if chart_type > len(get_supported_chart_types()) - 1 and not 99:
            print 'error: %s is not a valid chart type.\n' % chart_type
            print_help()
            sys.exit()
        os.system('%s %s' % (get_log_parsing_script(), sys.argv[1]))
        time.sleep(0.5)
        if chart_type == -1:
            if len(sys.argv) < 4:
                print 'Need output directory as fourth argument'
                sys.exit()
            else:
                num_fields = len(get_supported_chart_types())
                for i in range(0, num_fields):
                    plot_chart_save(i, sys.argv[1], sys.argv[3])
        elif chart_type == 99:
            plot_model(chart_type, sys.argv[1])
        else:
            plot_chart(chart_type, sys.argv[1])

    os.system("rm %s" % (sys.argv[1] + '.train'))
    os.system("rm %s" % (sys.argv[1] + '.test'))