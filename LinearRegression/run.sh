echo "Running batch gradient descent..."
python3 RunLinearRegression.py '../Data/concrete' batch 0.01
echo "Running stochastic gradient descent..."
python3 RunLinearRegression.py '../Data/concrete' stochastic 0.01
echo "Solving analyticaly..."
python3 GetAnalyticSolution.py '../Data/concrete'