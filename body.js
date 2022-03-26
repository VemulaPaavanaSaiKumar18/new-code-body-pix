var bodyPix = require("@tensorflow-models/body-pix");
var tfjs = require("@tensorflow/tfjs")
var inkjet = require('inkjet');
var createCanvas = require('canvas');
var fs = require('fs');

async function loadAndPredict(data) {
        const net = await bodyPix.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            multiplier: 0.75,
            quantBytes: 2
    });

    imgD = createCanvas.createImageData(new Uint8ClampedArray(data.data), data.width, data.height);
    const segmentation = await net.segmentPerson(imgD, {
    flipHorizontal: false,
    internalResolution: 'medium',
    segmentationThreshold: 0.7
    });
    const maskImage = bodyPix.toMask(segmentation, false);
}

inkjet.decode(fs.readFileSync('body.jfif'), function(err, data) {
    if (err) throw err;
    console.log('OK: Image');
    loadAndPredict(data)
  });