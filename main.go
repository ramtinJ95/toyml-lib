package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	dataDir    = "data"
	baseURL    = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	imageMagic = 2051
	labelMagic = 2049
)

var mnistFiles = []string{
	"train-images-idx3-ubyte.gz",
	"train-labels-idx1-ubyte.gz",
	"t10k-images-idx3-ubyte.gz",
	"t10k-labels-idx1-ubyte.gz",
}

func main() {
	rand.Seed(time.Now().UnixNano())

	if err := downloadMNIST(); err != nil {
		fmt.Printf("Error downloading MNIST: %v\n", err)
		return
	}

	fmt.Println("Loading training data...")
	trainImages, err := loadImages(filepath.Join(dataDir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		fmt.Printf("Error loading training images: %v\n", err)
		return
	}
	trainLabels, err := loadLabels(filepath.Join(dataDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		fmt.Printf("Error loading training labels: %v\n", err)
		return
	}

	fmt.Println("Loading test data...")
	testImages, err := loadImages(filepath.Join(dataDir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		fmt.Printf("Error loading test images: %v\n", err)
		return
	}
	testLabels, err := loadLabels(filepath.Join(dataDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		fmt.Printf("Error loading test labels: %v\n", err)
		return
	}

	fmt.Printf("Loaded %d training samples, %d test samples\n", len(trainImages), len(testImages))

	model := NewMLP(784, []int{128, 64, 10})
	params := model.Parameters()
	fmt.Printf("Model parameters: %d\n", len(params))

	epochs := 10
	learningRate := 0.01
	batchSize := 32
	trainSize := 10000 // Use subset for faster training

	fmt.Printf("\nTraining with %d samples, batch size %d, learning rate %.4f\n\n", trainSize, batchSize, learningRate)

	for epoch := 0; epoch < epochs; epoch++ {
		epochStart := time.Now()

		// Shuffle training indices
		indices := make([]int, trainSize)
		for i := range indices {
			indices[i] = i
		}
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		totalLoss := 0.0
		correct := 0
		numBatches := 0

		// Process mini-batches
		for i := 0; i < trainSize; i += batchSize {
			end := i + batchSize
			if end > trainSize {
				end = trainSize
			}

			batchLoss := NewValue(0.0)

			for j := i; j < end; j++ {
				idx := indices[j]

				// Convert image to Value inputs
				inputs := make([]*Value, 784)
				for k, pixel := range trainImages[idx] {
					inputs[k] = NewValue(pixel)
				}

				// Forward pass
				outputs := model.Forward(inputs)

				// Create one-hot target
				targets := oneHotEncode(trainLabels[idx], 10)

				// Compute MSE loss
				loss := MSE(outputs, targets)
				batchLoss = batchLoss.Add(loss)

				// Track accuracy
				if argmax(outputs) == trainLabels[idx] {
					correct++
				}
			}

			// Average loss over batch
			batchLoss = batchLoss.Mul(NewValue(1.0 / float64(end-i)))
			totalLoss += batchLoss.Data
			numBatches++

			// Zero gradients
			for _, p := range params {
				p.Grad = 0
			}

			// Backward pass
			batchLoss.Backprop()

			// SGD update
			for _, p := range params {
				p.Data -= learningRate * p.Grad
			}

			// Progress indicator every 50 batches
			if numBatches%50 == 0 {
				fmt.Printf("  Batch %d/%d\r", numBatches, trainSize/batchSize)
			}
		}

		// Calculate metrics
		avgLoss := totalLoss / float64(numBatches)
		trainAcc := float64(correct) / float64(trainSize) * 100

		// Evaluate on test set (use subset for speed)
		testSubset := 1000
		testAcc := evaluate(model, testImages[:testSubset], testLabels[:testSubset])

		epochTime := time.Since(epochStart)

		fmt.Printf("Epoch %d/%d | Loss: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%% | Time: %v\n",
			epoch+1, epochs, avgLoss, trainAcc, testAcc*100, epochTime.Round(time.Second))
	}

	// Final evaluation on full test set
	fmt.Println("\nFinal evaluation on full test set...")
	finalAcc := evaluate(model, testImages, testLabels)
	fmt.Printf("Final Test Accuracy: %.2f%%\n", finalAcc*100)
}

// downloadMNIST downloads the MNIST dataset if not already present
func downloadMNIST() error {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}

	for _, filename := range mnistFiles {
		path := filepath.Join(dataDir, filename)
		if _, err := os.Stat(path); err == nil {
			fmt.Printf("  %s already exists, skipping\n", filename)
			continue
		}

		fmt.Printf("  Downloading %s...\n", filename)
		url := baseURL + filename

		resp, err := http.Get(url)
		if err != nil {
			return fmt.Errorf("failed to download %s: %v", filename, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("failed to download %s: status %d", filename, resp.StatusCode)
		}

		file, err := os.Create(path)
		if err != nil {
			return err
		}
		defer file.Close()

		_, err = io.Copy(file, resp.Body)
		if err != nil {
			return err
		}
	}

	return nil
}

// loadImages reads MNIST image file and returns normalized float64 data
func loadImages(path string) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzReader.Close()

	var magic, numImages, numRows, numCols int32

	if err := binary.Read(gzReader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, fmt.Errorf("invalid magic number: got %d, expected %d", magic, imageMagic)
	}

	binary.Read(gzReader, binary.BigEndian, &numImages)
	binary.Read(gzReader, binary.BigEndian, &numRows)
	binary.Read(gzReader, binary.BigEndian, &numCols)

	pixelsPerImage := int(numRows * numCols)
	imageData := make([]byte, int(numImages)*pixelsPerImage)
	if _, err := io.ReadFull(gzReader, imageData); err != nil {
		return nil, err
	}

	images := make([][]float64, numImages)
	for i := int32(0); i < numImages; i++ {
		start := int(i) * pixelsPerImage
		images[i] = make([]float64, pixelsPerImage)
		for j := 0; j < pixelsPerImage; j++ {
			// Normalize to [0, 1]
			images[i][j] = float64(imageData[start+j]) / 255.0
		}
	}

	return images, nil
}

// loadLabels reads MNIST label file
func loadLabels(path string) ([]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzReader.Close()

	var magic, numLabels int32

	if err := binary.Read(gzReader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, fmt.Errorf("invalid magic number: got %d, expected %d", magic, labelMagic)
	}

	binary.Read(gzReader, binary.BigEndian, &numLabels)

	labelData := make([]byte, numLabels)
	if _, err := io.ReadFull(gzReader, labelData); err != nil {
		return nil, err
	}

	labels := make([]int, numLabels)
	for i, l := range labelData {
		labels[i] = int(l)
	}

	return labels, nil
}

// oneHotEncode converts a label to one-hot encoded Values
func oneHotEncode(label int, numClasses int) []*Value {
	encoded := make([]*Value, numClasses)
	for i := 0; i < numClasses; i++ {
		if i == label {
			encoded[i] = NewValue(1.0)
		} else {
			encoded[i] = NewValue(0.0)
		}
	}
	return encoded
}

// argmax returns the index of the maximum value
func argmax(outputs []*Value) int {
	maxIdx := 0
	maxVal := outputs[0].Data
	for i, v := range outputs {
		if v.Data > maxVal {
			maxVal = v.Data
			maxIdx = i
		}
	}
	return maxIdx
}

// evaluate computes accuracy on a dataset
func evaluate(model *MLP, images [][]float64, labels []int) float64 {
	correct := 0
	for i := range images {
		inputs := make([]*Value, 784)
		for j, pixel := range images[i] {
			inputs[j] = NewValue(pixel)
		}
		outputs := model.Forward(inputs)
		if argmax(outputs) == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(images))
}
