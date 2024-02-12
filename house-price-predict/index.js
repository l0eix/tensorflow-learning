import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
import csv from 'csv-parser';

export async function readCsvData() {
    const csvFilePath = './Housing.csv';
    let dataArray = [];

    fs.createReadStream(csvFilePath)
        .pipe(csv())
        .on('data', (row) => {
            dataArray.push(row);
        })
        .on('end', () => {
            dataArray = dataArray.map((item) => {
                return {
                    price: parseFloat(item.price),
                    area: parseFloat(item.area),
                    bedrooms: parseFloat(item.bedrooms),
                    bathrooms: parseFloat(item.bathrooms),
                    stories: parseFloat(item.stories),
                    parking: parseFloat(item.parking),
                    furnished: item.furnishingstatus === "furnished" ? 1 : 0,
                    unFurnished: item.furnishingstatus === "unfurnished" ? 1 : 0,
                    semiFurnished: item.furnishingstatus === "semi-furnished" ? 1 : 0,
                    mainroad: item.mainroad === "yes" ? 1 : 0,
                    guestroom: item.guestroom === "yes" ? 1 : 0,
                    hotwaterheating: item.hotwaterheating === "yes" ? 1 : 0,
                    airconditioning: item.airconditioning === "yes" ? 1 : 0,
                    prefarea: item.prefarea === "yes" ? 1 : 0,
                };
            });

            const trainingSet = dataArray.slice(0, dataArray.length / 2);
            const testingSet = dataArray.slice(dataArray.length / 2, dataArray.length);

            buildModel(trainingSet, testingSet);
        })
        .on('error', (error) => {
            console.error('Error reading CSV file:', error.message);
        });
}

export async function buildModel(trainingSet, testingSet) {
    // Extract features and labels from training set
    const trainingData = trainingSet.map((item) => {
        return [
            item.area,
            item.bedrooms,
            item.bathrooms,
            item.stories,
            item.parking,
            item.furnished,
            item.unFurnished,
            item.semiFurnished,
            item.mainroad,
            item.guestroom,
            item.hotwaterheating,
            item.airconditioning,
            item.prefarea
        ];
    });

    const trainingLabels = trainingSet.map((item) => item.price);

    // Extract features and labels from testing set
    const testingData = testingSet.map((item) => {
        return [
            item.area,
            item.bedrooms,
            item.bathrooms,
            item.stories,
            item.parking,
            item.furnished,
            item.unFurnished,
            item.semiFurnished,
            item.mainroad,
            item.guestroom,
            item.hotwaterheating,
            item.airconditioning,
            item.prefarea
        ];
    });

    const testingLabels = testingSet.map((item) => item.price);

    // Feature scaling: normalize the data
    const { dataMean, dataStd } = tf.tidy(() => {
        const tensorData = tf.tensor2d(trainingData);
        const dataMean = tensorData.mean(0);
        const dataStd = tensorData.sub(dataMean).square().mean(0).sqrt();

        return { dataMean, dataStd };
    });

    const normalizedTrainingData = tf.tensor2d(trainingData).sub(dataMean).div(dataStd);
    const normalizedTestingData = tf.tensor2d(testingData).sub(dataMean).div(dataStd);

    // Build the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [13] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Convert data to tensors
    const trainingDataTensor = normalizedTrainingData;
    const trainingLabelsTensor = tf.tensor1d(trainingLabels);

    // Train the model
    await model.fit(trainingDataTensor, trainingLabelsTensor, { epochs: 50 });

    // Evaluate the model on the testing set
    const evalResult = model.evaluate(normalizedTestingData, tf.tensor1d(testingLabels));
    console.log(`Mean Squared Error on Testing Set: ${evalResult.meanSquaredError}`);

    // Make predictions on a sample data point
    const sampleData = [
        7420,  // area
        4,     // bedrooms
        3,     // bathrooms
        2,     // stories
        1,     // parking
        0,     // furnished
        1,     // unFurnished
        0,     // semiFurnished
        1,     // mainroad
        0,     // guestroom
        1,     // hotwaterheating
        0,     // airconditioning
        0      // prefarea
    ];

    const normalizedSampleData = tf.tensor2d([sampleData]).sub(dataMean).div(dataStd);
    const predictionTensor = model.predict(normalizedSampleData);
    const predictedPrice = predictionTensor.dataSync()[0];

    console.log(`Predicted Price: ${predictedPrice}`);
}

readCsvData();