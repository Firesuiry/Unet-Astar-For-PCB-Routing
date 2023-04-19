// Initiate a canvas instance
var canvas = new fabric.Canvas("canvas");
canvas.setWidth(3000);
canvas.setHeight(3000);

var data = undefined;
var display_layer = 0;

function get_data() {
    fetch('http://127.0.0.1:5000/ ').then(res => {
        return res.json()
    }).then(res2 => {
        console.log(res2)
        data = res2
    }).then(res3 => {
        display_data()
    })
}

function display_data() {
    if (data === undefined) {
        return
    }

    for (var i = 0; i < data.pins.length; i++) {
        var pin = data.pins[i]
        var shape = pin['shape']
        padstack = data.padstacks[shape][display_layer+1]
        console.log(padstack)
        if (padstack === undefined) {
            continue
        } else {
            center = [pin['x'], pin['y']]
            detail = padstack['detail_p']
            if (padstack['shape'] === 'circle') {
                let radius = detail[0] / 2
                let delta_x = 0;
                let delta_y = 0;
                if (detail.length > 1) {
                    delta_x = detail[1]
                    delta_y = detail[2]
                }
                let x = center[0] + delta_x;
                let y = center[1] + delta_y;
                var element = new fabric.Circle({
                    radius: radius,
                    fill: 'red',
                    left: x,
                    top: y,
                    originX: 'center',
                    originY: 'center'
                });
                console.log(x,y,radius);
                canvas.add(element);
            } else if (padstack['shape'] === 'polygon') {
                continue
                var element = new fabric.Polygon(points, {
                    left: 0,
                    top: 0,
                    fill: "#1e90ff",
                    strokeWidth: 4,
                    stroke: "green",
                    flipY: true,
                    scaleX: 2,
                    scaleY: 2,
                    opacity: 0.5,
                });
                canvas.add(element);
            }
            element.on("mouseover", () => {
                console.log('pin' + i.toString() + ' mouseover')
                element.set("opacity", 1);
                canvas.renderAll();
            });
        }
    }
    canvas.renderAll();
}

// Initiating a points array
var points = [
    {x: 30, y: 50},
    {x: 0, y: 0},
    {x: 60, y: 0},
];

// Initiating a polygon object
var triangle = new fabric.Polygon(points, {
    left: 0,
    top: 0,
    fill: "#1e90ff",
    strokeWidth: 4,
    stroke: "green",
    flipY: true,
    scaleX: 2,
    scaleY: 2,
    opacity: 0.5,
});

// Adding it to the canvas
canvas.add(triangle);

// Using mouseover event
triangle.on("mouseover", () => {
    triangle.set("opacity", 1);
    canvas.renderAll();
});

// Using mouseout event
triangle.on("mouseout", () => {
    triangle.set("opacity", 0.5);
    canvas.renderAll();
});

get_data()
