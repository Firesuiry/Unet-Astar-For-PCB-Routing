// Initiate a canvas instance
var canvas = new fabric.Canvas("canvas");

var data = undefined;
var search_path = undefined;
var display_layer = 0;
var pin_bind = {};
var layer_element = document.getElementById("layer");
var msg_element = document.getElementById("msg");
var layer_input_element = document.getElementById("layer_num_input");
var net_input_element = document.getElementById("net_input");
var point_input_element = document.getElementById("point_input");
var multi_rate = 1;

canvas.on('mouse:down', function (e) {
    getMouseCoords(e);
});

function getMouseCoords(event) {
    var pointer = canvas.getPointer(event.e);
    var posX = pointer.x / multi_rate;
    var posY = pointer.y / multi_rate;
    console.log(posX + ", " + posY);    // Log to console
}

function get_data() {
    fetch('http://127.0.0.1:5000/ ').then(res => {
        return res.json()
    }).then(res2 => {
        console.log(res2)
        data = res2
    }).then(res3 => {
        display_data(0);
    })
}

function search_path_onclick() {
    var net_id = parseInt(net_input_element.value);
    get_search_path(net_id);
}

function get_search_path(net_id) {
    console.log('get_search_path' + net_id.toString())
    fetch('http://127.0.0.1:5000/search_path/' + net_id.toString()).then(res => {
        return res.json()
    }).then(res2 => {
        // console.log(res2)
        search_path = res2
    }).then(res3 => {
        // display_search_path(0);
        // display_data(0);
        for (var i = 0; i < point_element_list.length; i++) {
            canvas.remove(point_element_list[i]);
        }
        point_element_list = [];
        search_path_index = 0;
        search_path_animation();
    })
}

var search_path_index = 0;
var last_current_point_element = undefined;
var point_element_list = [];

function display_search_path(index = undefined) {
    if (index !== undefined) {
        search_path_index = index;
    } else {
        search_path_index += 1;
    }

    if (search_path_index >= search_path.length) {
        return;
    }

    current = search_path[search_path_index]['current'];
    neighbors = search_path[search_path_index]['neighbors'];
    if (last_current_point_element !== undefined) {
        canvas.remove(last_current_point_element);
    }
    last_current_point_element = new fabric.Circle({
        radius: 3,
        fill: 'red',
        left: current[1] * multi_rate,
        top: current[2] * multi_rate,
        lockMovementX: true,
        lockMovementY: true,
        selected: false,
    })
    canvas.add(last_current_point_element);
    for (var i = 0; i < neighbors.length; i++) {
        neighbor = neighbors[i];
        element = new fabric.Circle({
            radius: 2,
            fill: 'blue',
            left: neighbor[1] * multi_rate,
            top: neighbor[2] * multi_rate,
            lockMovementX: true,
            lockMovementY: true,
            selected: false,
            opacity: 0.5,
        })
        canvas.add(element);
        point_element_list.push(element);
    }

}

function search_path_animation() {
    if (search_path === undefined) {
        console.log('search_path is undefined')
        return
    }
    if (search_path_index >= search_path.length) {
        return;
    }

    display_search_path();
    setTimeout(search_path_animation, 100);
}

function display_data(active_layer) {
    canvas.clear();
    layer_element.textContent = active_layer.toString();
    if (data === undefined) {
        return
    }
    let max_length = Math.max(data.max_x, data.max_y);
    if (max_length < 500) {
        multi_rate = Math.ceil(1000 / max_length);
    } else {
        multi_rate = 1;
    }
    canvas.setWidth(data.max_x * multi_rate);
    canvas.setHeight(data.max_y * multi_rate);
    for (i = 0; i < data.steiner_nets.length; i++) {
        var net = data.steiner_nets[i]
        if (net.path === undefined) {
            continue
        } else {
            var path = net.path
            var points = []
            for (var j = 0; j < path.length; j++) {
                var point = path[j]
                points.push({
                    layer: point[0],
                    x: point[1] * multi_rate,
                    y: point[2] * multi_rate
                })
            }
            for (var j = 0; j < points.length - 1; j++) {
                start_point = points[j];
                end_point = points[j + 1];
                if (start_point.x === end_point.x && start_point.y === end_point.y) {
                    continue
                }
                element = new fabric.Line([start_point.x, start_point.y, end_point.x, end_point.y], {
                    stroke: start_point.layer === active_layer ? 'red' : 'blue',
                    strokeWidth: multi_rate,
                    opacity: 1,
                    lockMovementX: true,
                    lockMovementY: true,
                    selected: false,
                    evented: start_point.layer === active_layer,
                });
                element.id = i
                element.tp = 'net'
                element.old_id = net.old_index;
                element.on("mouseover", (e) => {
                    console.log(e.target.tp + e.target.id.toString() + ' mouseover')
                    e.target.set("opacity", 0.5);
                    canvas.renderAll();
                    t = e.target;
                    msg_element.textContent = t.tp + t.id.toString() + ':' + t.old_id + ' mouseover';
                })
                element.on("mouseout", (e) => {
                    e.target.set("opacity", 1);
                })
                canvas.add(element);
            }
        }

    }
    for (var current_layer = 0; current_layer < data.layer_num; current_layer++) {
        for (var i = 0; i < data.pins.length; i++) {
            var pin = data.pins[i]
            var shape = pin['shape']
            padstack = data.padstacks[shape][current_layer + 1]
            // console.log(padstack)
            if (padstack === undefined) {
                continue
            } else {
                color = current_layer === active_layer ? '#FFAA00' : '#003434'
                center = [pin['x'] * multi_rate, pin['y'] * multi_rate]
                detail = padstack['detail_p']
                if (padstack['shape'] === 'circle') {
                    let radius = detail[0] / 2 * multi_rate
                    let delta_x = 0;
                    let delta_y = 0;
                    if (detail.length > 1) {
                        delta_x = detail[1] * multi_rate
                        delta_y = detail[2] * multi_rate
                    }
                    let x = center[0] + delta_x;
                    let y = center[1] + delta_y;
                    var element = new fabric.Circle({
                        radius: radius,
                        fill: color,
                        left: x,
                        top: y,
                        originX: 'center',
                        originY: 'center',
                        evented: current_layer === active_layer,
                        opacity: 0.5,
                    });
                    // console.log(x, y, radius);
                    canvas.add(element);
                } else if (padstack['shape'] === 'polygon') {
                    detail = padstack['detail_p']
                    center = [pin['x'] * multi_rate, pin['y'] * multi_rate]
                    points = []
                    let minx = 9999
                    let miny = 9999;
                    for (var j = 0; j < detail.length / 2 - 1; j++) {
                        points.push({
                            x: detail[1 + 2 * j] * multi_rate,
                            y: detail[2 + 2 * j] * multi_rate
                        })
                        if (minx > detail[1 + 2 * j] * multi_rate) {
                            minx = detail[1 + 2 * j] * multi_rate
                        }
                        if (miny > detail[2 + 2 * j] * multi_rate) {
                            miny = detail[2 + 2 * j] * multi_rate
                        }
                    }
                    var element = new fabric.Polygon(points, {
                        left: pin['x'] * multi_rate + minx,
                        top: pin['y'] * multi_rate + miny,
                        fill: color,
                        strokeWidth: 0,
                        stroke: "green",
                        flipY: true,
                        scaleX: 1,
                        scaleY: 1,
                        opacity: 0.5,
                        evented: current_layer === active_layer,
                    });
                    canvas.add(element);
                }
                element.id = i
                element.tp = 'pin'
                element.on("mouseover", (e) => {
                    console.log(e.target);
                    console.log(element.tp + e.target.id.toString() + ' mouseover')
                    element.set("opacity", 1);
                    msg_element.textContent = e.target.tp + e.target.id.toString() + '|' + data.pins[e.target.id].layers;
                });
            }
        }
    }
    canvas.renderAll();
}

function change_layer() {
    let layer_num = parseInt(layer_input_element.value);
    display_data(layer_num);
}

function show_point() {
    let point_str = point_input_element.value;
    let point_ids = point_str.split(',');
    var element = new fabric.Circle({
        radius: 1 * multi_rate,
        fill: '#FF00FF',
        left: point_ids[0] * multi_rate,
        top: point_ids[1] * multi_rate,
        originX: 'center',
        originY: 'center',
        evented: false,
        opacity: 1,
    });
    canvas.add(element);

    canvas.renderAll();
}

function show_net() {
    let net_str = net_input_element.value;
    let net_ids = net_str.split(',');
    for (let i = 0; i < net_ids.length; i++) {
        let net_id = parseInt(net_ids[i]);
        console.log('show net ' + net_id)
        var net = data.steiner_nets[net_id]
        if (net === undefined) {
            continue
        }
        if (net.path === undefined) {
            continue
        }
        var path = net.path
        var points = []
        for (var j = 0; j < path.length; j++) {
            var point = path[j]
            points.push({
                layer: point[0] * multi_rate,
                x: point[1] * multi_rate,
                y: point[2] * multi_rate
            })
        }
        for (var j = 0; j < points.length - 1; j++) {
            start_point = points[j];
            end_point = points[j + 1];

            element = new fabric.Line([start_point.x, start_point.y, end_point.x, end_point.y], {
                stroke: 'green',
                opacity: 1,
                lockMovementX: true,
                lockMovementY: true,
                selected: false,
                strokeWidth: 1,
            });
            element.id = i
            element.tp = 'net'
            element.old_id = net.old_index;
            element.on("mouseover", (e) => {
                console.log(e.target.tp + e.target.id.toString() + ' mouseover')
                e.target.set("opacity", 0.5);
                canvas.renderAll();
                t = e.target;
                msg_element.textContent = t.tp + t.id.toString() + ':' + t.old_id + ' mouseover';
            })
            element.on("mouseout", (e) => {
                e.target.set("opacity", 1);
            })
            canvas.add(element);
        }
    }


}

get_data()
// var int=self.setInterval("get_data()",100000);
