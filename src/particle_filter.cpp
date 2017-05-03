/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>

#include "particle_filter.h"
using namespace std;

static default_random_engine gen;

static LandmarkObs transformObservation(double partX, double partY, double partTheta,
                                                   double observX, double observY) {
  double obsNewX = observX * cos(partTheta) - observY * sin(partTheta) + partX;
  double obsNewY = observX * sin(partTheta) + observY * cos(partTheta) + partY;

  return LandmarkObs{0, obsNewX, obsNewY};
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 500;
  weights.resize(num_particles);
  particles.resize(num_particles);

  normal_distribution<double> N_x_init(0, std[0]);
  normal_distribution<double> N_y_init(0, std[1]);
  normal_distribution<double> N_theta_init(0, std[2]);
  double n_x;
  double n_y;
  double n_theta;

  for (int i=0; i<num_particles; i++) {
    weights[i] = 1.0;

    n_x = N_x_init(gen);
    n_y = N_y_init(gen);
    n_theta = N_theta_init(gen);

    particles[i] = Particle {i, x + n_x, y + n_y, theta + n_theta, weights[i]};
  }

  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> N_x_init(0, std_pos[0]);
  normal_distribution<double> N_y_init(0, std_pos[1]);
  normal_distribution<double> N_theta_init(0, std_pos[2]);

  for (int i=0; i<num_particles; i++) {
    double n_x = N_x_init(gen);
    double n_y = N_y_init(gen);
    double n_theta = N_theta_init(gen);

    // update heading
    particles[i].theta += yaw_rate * delta_t + n_theta;

    // move in the (noisy) commanded direction
    double dist_x = velocity * delta_t + n_x;
    particles[i].x += cos(particles[i].theta) * dist_x;

    double dist_y = velocity * delta_t + n_y;
    particles[i].y += sin(particles[i].theta) * dist_y;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i=0; i<observations.size(); i++) {
    double min;
    int closestLandmarkId = -1;

    for (int j=0; j<predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (j == 0 || distance < min) {
        min = distance;
        closestLandmarkId = predicted[j].id;
      }
    }

    observations[i].id = closestLandmarkId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

  // prepare convenient map: id->reference to landmark

  map<int,Map::single_landmark_s*> idx2lm;
  for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
    idx2lm.insert(pair<int,Map::single_landmark_s*>(map_landmarks.landmark_list[i].id_i, &map_landmarks.landmark_list[i]));
  }

  // prepare helper variables for calculating weights

  const double std_x = std_landmark[0];
  const double std_y = std_landmark[1];
  const double c = 1/(2*M_PI * sqrt(std_x) * sqrt(std_y));

  // iterate over particles

  for (int k=0; k<num_particles; k++) {
    double partX = particles[k].x;
    double partY = particles[k].y;
    double partTheta = particles[k].theta;

    // predict landmarks - only ones within the range

    vector<LandmarkObs> predicted;
    for (int h=0; h<map_landmarks.landmark_list.size(); h++) {
      int id = map_landmarks.landmark_list[h].id_i;
      double landX = map_landmarks.landmark_list[h].x_f;
      double landY = map_landmarks.landmark_list[h].y_f;

      if (dist(partX, partY, landX, landY) < sensor_range) {
        predicted.push_back(LandmarkObs{id, landX, landY});
      }
    }

    // measurements form local to global

    vector<LandmarkObs> transformedObs;
    for (int i=0; i < observations.size(); i++) {
      double obsX = observations[i].x;
      double obsY = observations[i].y;

      LandmarkObs obs = transformObservation(partX, partY, partTheta, obsX, obsY);
      transformedObs.push_back(obs);
    }

    // find nearest landmarks

    dataAssociation(predicted, transformedObs);

    // calculate probabilities

    double prob = 1.0;

    for (int i=0; i<transformedObs.size(); i++) {

      // if no matching landmark then assign 0 prob for the particle
      if (transformedObs[i].id == -1) {
        prob = 0.0;
        break;

      // calculate new prob for the particle
      } else {
        Map::single_landmark_s *lm;
        lm = idx2lm[transformedObs[i].id];
        double x_lm = lm->x_f;
        double y_lm = lm->y_f;
        double x_obs = transformedObs[i].x;
        double y_obs = transformedObs[i].y;

        double x_diff = pow(x_obs - x_lm, 2)/std_x;
        double y_diff = pow(y_obs - y_lm, 2)/std_y;

        prob *= c*exp(-( x_diff + y_diff )/2);
      }
    }

    particles[k].weight = prob;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // prepare vector of weights

  for (int i=0; i<particles.size(); i++) {
    weights[i] = particles[i].weight;
  }

  // draw new set of particles

  discrete_distribution<> distribution(weights.begin(), weights.end());

  vector<Particle> resample;
  for (int i=0; i<particles.size(); i++) {
    int weighted_index = distribution(gen);
    resample.push_back(particles[weighted_index]);
  }

  particles = resample;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
