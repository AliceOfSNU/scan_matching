using namespace std;

#include <string>
#include <sstream>
#include "helper.h"

using namespace Eigen;
#include <pcl/registration/icp.h>
#include <pcl/console/time.h>   // TicToc

Pose pose(Point(0,0,0),Rotate(0,0,0));
Pose upose = pose;
bool matching = false;
bool update = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

	if (event.getKeySym() == "space" && event.keyDown()){
		matching = true;
		
  	}

}

double Probability(Eigen::MatrixXd X, Eigen::MatrixXd Q, Eigen::MatrixXd S){
  	Eigen::MatrixXd V = (X-Q).transpose() * S.inverse() * (X-Q);
  	double x = V(0, 0);
  	
	return exp(-0.5*x);
}

struct Cell{
	PointCloudT::Ptr cloud;
	Eigen::MatrixXd Q;
	Eigen::MatrixXd S;

	Cell(){
		PointCloudT::Ptr input(new PointCloudT);
		cloud = input;
		Q = Eigen::MatrixXd::Zero(2,1);
		S = Eigen::MatrixXd::Zero(2,2);
	}
};

struct Grid{

	// each cell is a square res x res and total grid size is (2 x width) x (2 x height)
	double res;
	int width;
	int height;

	vector<vector<Cell>> grid;

	Grid(double setRes, int setWidth, int setHeight){

		res = setRes;
		width = setWidth;
		height = setHeight;

		for(int r = 0; r < height*2; r++ ){
			vector<Cell> row;
			for(int c = 0; c < width*2; c++){
				row.push_back(Cell());
			}
			grid.push_back(row);
		}
	}

	void addPoint(PointT point){

		//cout << point.x << "," << point.y << endl;

		int c = int((point.x + width * res  ) / res);
		int r = int((point.y + height * res ) / res);

		//cout << r << "," << c << endl;

		if( (c >= 0 && c < width*2) && (r >= 0 && r < height*2) ){
			grid[r][c].cloud->points.push_back(point);
		}
	} 

	void Build(){

		for(int r = 0; r < height*2; r++ ){
			for(int c = 0; c < width*2; c++){

				PointCloudT::Ptr input = grid[r][c].cloud;
				if(input->points.size() > 2){

					// Calculate the mean
					Eigen::MatrixXd Q(2,1);
					Q << Eigen::MatrixXd::Zero(2,1);
					for(PointT point : input->points){
						Q(0,0) += point.x;
						Q(1,0) += point.y;
					}
					Q(0,0) = Q(0,0)/input->points.size();
					Q(1,0) = Q(1,0)/input->points.size();

					grid[r][c].Q = Q;

					// Calculate sigma
					Eigen::MatrixXd S(2,2);
					S << Eigen::MatrixXd::Zero(2,2);
					for(PointT point : input->points){
						Eigen::MatrixXd X(2,1);
						X(0,0) = point.x;
						X(1,0) = point.y;

						S += (X-Q) * (X-Q).transpose();
					}
					S(0,0) = S(0,0)/input->points.size();
					S(0,1) = S(0,1)/input->points.size();
					S(1,0) = S(1,0)/input->points.size();
					S(1,1) = S(1,1)/input->points.size();

					grid[r][c].S = S;
				}
				
			}
		}
	}

	Cell getCell(PointT point){
		int c = int((point.x + width * res  ) / res);
		int r = int((point.y + height * res ) / res);

		if( (c >= 0 && c < width*2) && (r >= 0 && r < height*2) ){
			return grid[r][c];
		}
		return Cell();
	}

	double Value(PointT point){

		Eigen::MatrixXd X(2,1);
		X(0,0) = point.x;
		X(1,0) = point.y;

		double value = 0;
		for(int r = 0; r < height*2; r++ ){
			for(int c = 0; c < width*2; c++){
				if(grid[r][c].cloud->points.size() > 2)
					value += Probability(X, grid[r][c].Q, grid[r][c].S );
			}
		}
		return value;
	}
};

Cell PDF(PointCloudT::Ptr input, int res, pcl::visualization::PCLVisualizer::Ptr& viewer){

	Eigen::MatrixXd Q(2,1);
	Q << Eigen::MatrixXd::Zero(2,1);
	
  
  	double sx = 0.0; double sy = 0.0;
  	int N = input->points.size();
  
  	// data
  	Eigen::MatrixXd coords = Eigen::MatrixXd(N, 2);
  	coords << Eigen::MatrixXd::Zero(N, 2);
  
  	for(int i = 0; i < N; ++i){
      PointT p = input->points[i];
      sx += p.x; sy += p.y;
      coords(i, 0) = p.x; coords(i, 1) = p.y;
    }
	Q(0, 0) = sx/N; Q(1, 0) = sy/N;
  	
	Eigen::MatrixXd S(2,2);
	S << Eigen::MatrixXd::Zero(2,2);
	
  	for(int i = 0; i < N; ++i){
      for(int j = 0; j < 2; ++j){
        coords(i, j) -= Q(j, 0);
      }
    }
	S = coords.transpose() * coords / N;
  
	PointCloudTI::Ptr pdf(new PointCloudTI);
	for(double i = 0.0; i <= 10.0; i += 10.0/double(res)){
		for(double j = 0.0; j <= 10.0; j += 10.0/double(res)){
			Eigen::MatrixXd X(2,1);
			X(0,0) = i;
			X(1,0) = j;

			PointTI point;
			point.x = i;
			point.y = j;
			point.z = Probability(X,Q,S);
			point.intensity = point.z;
			pdf->points.push_back(point);
		}
	}
	renderPointCloudI(viewer, pdf, "pdf");

	Cell cell = Cell();
	cell.S = S;
	cell.Q = Q;
	cell.cloud = input;
	return cell;
}

template<typename Derived>
void NewtonsMethod(PointT point, double theta, Cell cell, Eigen::MatrixBase<Derived>& g_previous, Eigen:: MatrixBase<Derived>& H_previous){
	double x = point.x; double y = point.y;
	Eigen::MatrixXd Q = cell.Q;
    Eigen::MatrixXd SINV = cell.S.inverse();
    
	Eigen::MatrixXd X = Eigen::MatrixXd(2,1);
    X(0, 0) = point.x; X(1, 0) = point.y;
    
	Eigen::MatrixXd q = X - Q;
    
	Eigen::MatrixXd dqdx(2,1), dqdy(2,1), dqdth(2,1);
    dqdx(0, 0) = 1.0; dqdx(1, 0) = 0.0;
    dqdy(0, 0) = 0.0; dqdy(1, 0) = 1.0;
    double st = sin(theta);
    double ct = cos(theta);
    dqdth(0, 0) = -x*st -y*ct;
    dqdth(1, 0) = x*ct - y*st;
    
    vector<Eigen::MatrixXd> dqs {dqdx, dqdy, dqdth};
    
	Eigen::MatrixXd EXP(1,1);
    EXP(0,0) = exp(-0.5*(q.transpose() * SINV * q)(0,0));
    
	Eigen::MatrixXd g(3,1);
	g << Eigen::MatrixXd::Zero(3,1);
    g(0, 0) = ( q.transpose()*SINV*dqdx*EXP )(0,0);
    g(1, 0) = ( q.transpose()*SINV*dqdy*EXP )(0,0);
    g(2, 0) = (q.transpose()*SINV*dqdth*EXP )(0,0);
    
	Eigen::MatrixXd ddq(2,1);
    ddq(0,0) = -x*ct + y*st;
    ddq(1,0) = -x*st - y*ct;
	Eigen::MatrixXd H(3,3);
	H << Eigen::MatrixXd::Zero(3,3);
    for(int i = 0; i < 3; ++i){
    	for(int j = 0; j < 3; ++j){
        	H(i, j) = (-EXP*(
            (-q.transpose()*SINV*dqs[i])*(-q.transpose()*SINV*dqs[j]) 
            + (-dqs[j].transpose()*SINV*dqs[i])))(0, 0);
        }
    }
    H(2,2) = (-EXP*(
            (-q.transpose()*SINV*dqs[2])*(-q.transpose()*SINV*dqs[2]) 
            + (-q.transpose()*SINV*ddq)+(-dqs[2].transpose()*SINV*dqs[2])))(0, 0);
	
	//cout << "H" << endl << H << "g" << endl << g << endl;
    //cout << "iter ..... " << endl;
	H_previous += H;
	g_previous += g;

}
double Score(PointCloudT::Ptr cloud, Grid grid){
	double score = 0;
	for(PointT point:cloud->points){
		Cell cell = grid.getCell(point);
		if(cell.cloud->points.size() > 2){
			score += grid.Value(point);
		}
	}
	return score;
}

double AdjustmentScore(double alpha, Eigen::MatrixXd T, PointCloudT::Ptr source, Pose pose, Grid grid){

	T *= alpha;
	pose.position.x += T(0,0);
	pose.position.y += T(1,0);
	pose.rotation.yaw += T(2,0);
	while(pose.rotation.yaw > 2*pi)
		pose.rotation.yaw -= 2*pi;

	
	Eigen::Matrix4d transform = transform3D(pose.rotation.yaw, 0, 0, pose.position.x, pose.position.y, 0);

	PointCloudT::Ptr transformed_scan (new PointCloudT);
	pcl::transformPointCloud (*source, *transformed_scan, transform);

	double score = Score(transformed_scan, grid);

	//cout << "score would be " << score << endl;

	return score;

}

double computeStepLength(Eigen::MatrixXd T, PointCloudT::Ptr source, Pose pose, Grid grid, double currScore){
	double maxParam = max( max( T(0,0), T(1,0)), T(2,0) );
	double mlength = 1.0;
	if(maxParam > 0.2){
		mlength =  0.1/maxParam;
		T *= mlength;
	}

	double bestAlpha = 0;

	//Try smaller steps
	double alpha = 1.0;
	for(int i = 0; i < 40; i++){
		double adjScore = AdjustmentScore(alpha, T, source, pose, grid);
		if( adjScore > currScore){
			bestAlpha = alpha;
			currScore = adjScore;
		}
		alpha *= .7;
	}
	if(bestAlpha == 0){
		//Try larger steps
		alpha = 2.0;
		for(int i = 0; i < 10; i++){
			double adjScore = AdjustmentScore(alpha, T, source, pose, grid);
			if( adjScore > currScore){
				bestAlpha = alpha;
				currScore = adjScore;
			}
			alpha *= 2;
		}
	}

	return bestAlpha * mlength;
}

template<typename Derived>
double PosDef(Eigen::MatrixBase<Derived>& A, double start, double increment, int maxIt){

	bool pass = false;
	int count = 0;

	A += start * Eigen::Matrix3d::Identity ();
	
	while(!pass && count < maxIt){

		Eigen::LLT<Eigen::MatrixXd> lltOfA(A); // compute the Cholesky decomposition of A

    	if(lltOfA.info() == Eigen::NumericalIssue){
			A += increment * Eigen::Matrix3d::Identity ();
			count++;
    	}
		else{
			pass = true;
		}
	}

	return  start + increment * count;
}

int main(){

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("NDT Creation"));
  	viewer->setBackgroundColor (0, 0, 0);
  	viewer->addCoordinateSystem (1.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);


	// Load target
	PointCloudT::Ptr target(new PointCloudT);
	pcl::io::loadPCDFile("target.pcd", *target);

	renderPointCloud(viewer, target, "target", Color(0,0,1));
	
	Grid ndtGrid(3.0, 2, 2); 
	
	for(PointT point : target->points){
		ndtGrid.addPoint(point);
	}
	ndtGrid.Build();
	
	// Draw grid
	int rowc = 0;
	for(double y = -ndtGrid.height * ndtGrid.res; y <= ndtGrid.height * ndtGrid.res; y += ndtGrid.res){
		renderRay(viewer, PointT(-ndtGrid.width * ndtGrid.res,y,0), PointT(ndtGrid.width * ndtGrid.res,y,0), "grid_row_"+to_string(rowc), Color(0,0,1));
		rowc++;
	}
	int colc = 0;
	for(double x = -ndtGrid.width * ndtGrid.res; x <= ndtGrid.width * ndtGrid.res; x += ndtGrid.res){
		renderRay(viewer, PointT(x,-ndtGrid.height * ndtGrid.res,0), PointT(x,ndtGrid.height * ndtGrid.res,0), "grid_col_"+to_string(colc), Color(0,0,1));
		colc++;
	}

	// Draw total PDF from all cells
	PointCloudTI::Ptr pdf(new PointCloudTI);
	int res = 10;
	for(double y = -ndtGrid.height * ndtGrid.res; y <= ndtGrid.height * ndtGrid.res; y += ndtGrid.res/double(res)){
		for(double x = -ndtGrid.width * ndtGrid.res; x <= ndtGrid.width * ndtGrid.res; x += ndtGrid.res/double(res)){
			Eigen::MatrixXd X(2,1);
			X(0,0) = x;
			X(1,0) = y;

			PointTI point;
			point.x = x; point.y = y;
			double value = ndtGrid.Value(PointT(x,y,0));
			point.z = value;
			point.intensity = value;
			if(value > 0.01)
				pdf->points.push_back(point);
		}
	}
	
	renderPointCloudI(viewer, pdf, "pdf");

	// Load source
	PointCloudT::Ptr source(new PointCloudT);
	pcl::io::loadPCDFile("source.pcd", *source);

	renderPointCloud(viewer, source, "source", Color(1,0,0));

	double sourceScore = Score(source, ndtGrid);
	
	viewer->addText("Score: "+to_string(sourceScore), 200, 200, 32, 1.0, 1.0, 1.0, "score",0);

	double currentScore = sourceScore;

	int iteration = 0;
	while (!viewer->wasStopped ())
	{

		if(matching){

			viewer->removeShape("score");
			viewer->addText("Score: "+to_string(currentScore), 200, 200, 32, 1.0, 1.0, 1.0, "score",0);

			Eigen::MatrixXd g(3,1);
			g << Eigen::MatrixXd::Zero(3,1);

			Eigen::MatrixXd H(3,3);
			H << Eigen::MatrixXd::Zero(3,3);

			for(PointT point:source->points){
				Cell cell = ndtGrid.getCell(point);
				if(cell.cloud->points.size() > 2){
					double theta = pose.rotation.yaw;
					double x = pose.position.x;
					double y = pose.position.y;

					PointT pointTran(point.x*cos(theta)-point.y*sin(theta)+x, point.x*sin(theta)+point.y*cos(theta)+y, point.z);
					NewtonsMethod(pointTran, theta, cell, g, H);
				}
			}
			
			PosDef(H, 0, 5, 100); 

			Eigen::MatrixXd T = -H.inverse()*g;

			double alpha = computeStepLength(T, source, pose, ndtGrid, currentScore);

			T *= alpha;

			pose.position.x += T(0,0);
			pose.position.y += T(1,0);
			pose.rotation.yaw += T(2,0);
			while(pose.rotation.yaw > 2*pi)
				pose.rotation.yaw -= 2*pi;
			
			// calculate score using source transformed by the updated pose
			Eigen::Matrix4d transform = transform3D(pose.rotation.yaw, 0, 0, pose.position.x, pose.position.y, 0);

			PointCloudT::Ptr transformed_scan (new PointCloudT);
			pcl::transformPointCloud (*source, *transformed_scan, transform);

			double ndtScore = Score(transformed_scan, ndtGrid);
			
			viewer->removeShape("nscore");
			viewer->addText("NDT Score: "+to_string(ndtScore), 200, 150, 32, 1.0, 1.0, 1.0, "nscore",0);
			currentScore = ndtScore;
			iteration++;

			viewer->removePointCloud("ndt_scan");
			renderPointCloud(viewer, transformed_scan, "ndt_scan", Color(0,1,0));

			matching = false;
		}
		else if(update){

			Eigen::Matrix4d userTransform = transform3D(upose.rotation.yaw, upose.rotation.pitch, upose.rotation.roll, upose.position.x, upose.position.y, upose.position.z);

			PointCloudT::Ptr transformed_scan (new PointCloudT);
			pcl::transformPointCloud (*source, *transformed_scan, userTransform);
			viewer->removePointCloud("usource");
			renderPointCloud(viewer, transformed_scan, "usource", Color(0,1,1));

			
			double score = Score(transformed_scan, ndtGrid);
			viewer->removeShape("score");
			viewer->addText("Score: "+to_string(score), 200, 200, 32, 1.0, 1.0, 1.0, "score",0);
			
			update = false;
			
		}
		
		viewer->spinOnce ();
	}

	return 0;
}
