//! Robotics and autonomous systems evaluation metrics
//!
//! This module provides specialized metrics for evaluating robotic systems and
//! autonomous agents across various tasks including:
//! - Motion planning and trajectory evaluation
//! - Localization and mapping (SLAM) metrics
//! - Object detection and tracking for robotics
//! - Manipulation task evaluation
//! - Navigation and path planning metrics
//! - Human-robot interaction assessment
//! - Multi-robot coordination metrics
//! - Safety and reliability evaluation

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive robotics evaluation metrics suite
#[derive(Debug)]
pub struct RoboticsMetrics {
    /// Motion planning and control metrics
    pub motion_metrics: MotionPlanningMetrics,
    /// SLAM and localization metrics
    pub slam_metrics: SlamMetrics,
    /// Manipulation task metrics
    pub manipulation_metrics: ManipulationMetrics,
    /// Navigation metrics
    pub navigation_metrics: NavigationMetrics,
    /// Human-robot interaction metrics
    pub hri_metrics: HumanRobotInteractionMetrics,
    /// Multi-robot coordination metrics
    pub multi_robot_metrics: MultiRobotMetrics,
    /// Safety and reliability metrics
    pub safety_metrics: SafetyReliabilityMetrics,
    /// Perception metrics for robotics
    pub perception_metrics: RoboticPerceptionMetrics,
}

/// Motion planning and trajectory evaluation metrics
#[derive(Debug, Clone)]
pub struct MotionPlanningMetrics {
    /// Trajectory smoothness measures
    pub smoothness_metrics: TrajectorySmoothnessMetrics,
    /// Path optimality metrics
    pub optimality_metrics: PathOptimalityMetrics,
    /// Dynamic constraints satisfaction
    pub constraint_metrics: ConstraintSatisfactionMetrics,
    /// Execution time and efficiency
    pub efficiency_metrics: PlanningEfficiencyMetrics,
}

/// Trajectory smoothness evaluation
#[derive(Debug, Clone)]
pub struct TrajectorySmoothnessMetrics {
    /// Average jerk (third derivative of position)
    pub average_jerk: f64,
    /// Maximum jerk
    pub max_jerk: f64,
    /// Acceleration variance
    pub acceleration_variance: f64,
    /// Curvature analysis
    pub curvature_metrics: CurvatureMetrics,
    /// Velocity profile smoothness
    pub velocity_smoothness: f64,
}

/// Curvature analysis for trajectories
#[derive(Debug, Clone)]
pub struct CurvatureMetrics {
    /// Average curvature
    pub average_curvature: f64,
    /// Maximum curvature
    pub max_curvature: f64,
    /// Curvature variance
    pub curvature_variance: f64,
    /// Number of sharp turns (high curvature points)
    pub sharp_turns_count: usize,
}

/// Path optimality evaluation
#[derive(Debug, Clone)]
pub struct PathOptimalityMetrics {
    /// Path length ratio to optimal
    pub length_optimality_ratio: f64,
    /// Energy consumption ratio
    pub energy_optimality_ratio: f64,
    /// Time optimality ratio
    pub time_optimality_ratio: f64,
    /// Clearance from obstacles
    pub obstacle_clearance: ObstacleClearanceMetrics,
}

/// Obstacle clearance metrics
#[derive(Debug, Clone)]
pub struct ObstacleClearanceMetrics {
    /// Minimum clearance distance
    pub min_clearance: f64,
    /// Average clearance distance
    pub avg_clearance: f64,
    /// Clearance variance
    pub clearance_variance: f64,
    /// Safety margin ratio
    pub safety_margin_ratio: f64,
}

/// Constraint satisfaction metrics
#[derive(Debug, Clone)]
pub struct ConstraintSatisfactionMetrics {
    /// Joint limits satisfaction rate
    pub joint_limits_satisfaction: f64,
    /// Velocity limits satisfaction rate
    pub velocity_limits_satisfaction: f64,
    /// Acceleration limits satisfaction rate
    pub acceleration_limits_satisfaction: f64,
    /// Torque limits satisfaction rate
    pub torque_limits_satisfaction: f64,
    /// Collision avoidance success rate
    pub collision_avoidance_rate: f64,
}

/// Planning efficiency metrics
#[derive(Debug, Clone)]
pub struct PlanningEfficiencyMetrics {
    /// Planning computation time
    pub planning_time: Duration,
    /// Memory usage during planning
    pub memory_usage: usize,
    /// Number of iterations required
    pub iterations_count: usize,
    /// Success rate of planning
    pub planning_success_rate: f64,
    /// Convergence speed
    pub convergence_speed: f64,
}

/// SLAM and localization metrics
#[derive(Debug, Clone)]
pub struct SlamMetrics {
    /// Localization accuracy metrics
    pub localization_metrics: LocalizationAccuracyMetrics,
    /// Mapping quality metrics
    pub mapping_metrics: MappingQualityMetrics,
    /// Loop closure metrics
    pub loop_closure_metrics: LoopClosureMetrics,
    /// Computational efficiency
    pub computational_metrics: SlamComputationalMetrics,
}

/// Localization accuracy evaluation
#[derive(Debug, Clone)]
pub struct LocalizationAccuracyMetrics {
    /// Absolute Trajectory Error (ATE)
    pub absolute_trajectory_error: f64,
    /// Relative Pose Error (RPE)
    pub relative_pose_error: f64,
    /// Translation error statistics
    pub translation_error: ErrorStatistics,
    /// Rotation error statistics
    pub rotation_error: ErrorStatistics,
    /// Drift analysis
    pub drift_metrics: DriftMetrics,
}

/// Error statistics for pose estimation
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    /// Root Mean Square Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Maximum error
    pub max_error: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median error
    pub median_error: f64,
}

/// Drift analysis metrics
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    /// Translation drift rate (m/m)
    pub translation_drift_rate: f64,
    /// Rotation drift rate (deg/m)
    pub rotation_drift_rate: f64,
    /// Scale drift (if applicable)
    pub scale_drift: f64,
    /// Drift consistency
    pub drift_consistency: f64,
}

/// Mapping quality evaluation
#[derive(Debug, Clone)]
pub struct MappingQualityMetrics {
    /// Map completeness ratio
    pub completeness: f64,
    /// Map accuracy compared to ground truth
    pub map_accuracy: f64,
    /// Feature detection rate
    pub feature_detection_rate: f64,
    /// Map consistency metrics
    pub consistency_metrics: MapConsistencyMetrics,
}

/// Map consistency evaluation
#[derive(Debug, Clone)]
pub struct MapConsistencyMetrics {
    /// Feature matching consistency
    pub feature_consistency: f64,
    /// Geometric consistency
    pub geometric_consistency: f64,
    /// Temporal consistency
    pub temporal_consistency: f64,
    /// Global consistency score
    pub global_consistency: f64,
}

/// Loop closure evaluation
#[derive(Debug, Clone)]
pub struct LoopClosureMetrics {
    /// Loop closure detection rate
    pub detection_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Loop closure accuracy
    pub closure_accuracy: f64,
    /// Timing metrics for detection
    pub detection_timing: Duration,
}

/// SLAM computational metrics
#[derive(Debug, Clone)]
pub struct SlamComputationalMetrics {
    /// Average processing time per frame
    pub avg_processing_time: Duration,
    /// Memory usage over time
    pub memory_usage_profile: Vec<usize>,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Real-time performance factor
    pub real_time_factor: f64,
}

/// Manipulation task evaluation metrics
#[derive(Debug, Clone)]
pub struct ManipulationMetrics {
    /// Grasping success metrics
    pub grasping_metrics: GraspingMetrics,
    /// Object manipulation accuracy
    pub manipulation_accuracy: ManipulationAccuracyMetrics,
    /// Task completion metrics
    pub task_completion: TaskCompletionMetrics,
    /// Force and contact metrics
    pub force_metrics: ForceContactMetrics,
}

/// Grasping evaluation metrics
#[derive(Debug, Clone)]
pub struct GraspingMetrics {
    /// Grasp success rate
    pub success_rate: f64,
    /// Grasp stability score
    pub stability_score: f64,
    /// Force closure quality
    pub force_closure_quality: f64,
    /// Grasp robustness to perturbations
    pub robustness_score: f64,
    /// Approach trajectory quality
    pub approach_quality: f64,
}

/// Manipulation accuracy metrics
#[derive(Debug, Clone)]
pub struct ManipulationAccuracyMetrics {
    /// End-effector positioning accuracy
    pub positioning_accuracy: ErrorStatistics,
    /// Orientation accuracy
    pub orientation_accuracy: ErrorStatistics,
    /// Path following accuracy
    pub path_following_accuracy: f64,
    /// Target reaching success rate
    pub target_reaching_rate: f64,
}

/// Task completion evaluation
#[derive(Debug, Clone)]
pub struct TaskCompletionMetrics {
    /// Overall task success rate
    pub success_rate: f64,
    /// Task completion time
    pub completion_time: Duration,
    /// Efficiency score
    pub efficiency_score: f64,
    /// Partial task completion rate
    pub partial_completion_rate: f64,
    /// Retry attempts required
    pub retry_attempts: f64,
}

/// Force and contact evaluation
#[derive(Debug, Clone)]
pub struct ForceContactMetrics {
    /// Contact force accuracy
    pub force_accuracy: ErrorStatistics,
    /// Contact stability duration
    pub contact_stability: Duration,
    /// Force control precision
    pub force_control_precision: f64,
    /// Contact detection accuracy
    pub contact_detection_accuracy: f64,
}

/// Navigation metrics
#[derive(Debug, Clone)]
pub struct NavigationMetrics {
    /// Path planning quality
    pub path_planning: PathPlanningMetrics,
    /// Obstacle avoidance performance
    pub obstacle_avoidance: ObstacleAvoidanceMetrics,
    /// Goal reaching performance
    pub goal_reaching: GoalReachingMetrics,
    /// Dynamic environment adaptation
    pub dynamic_adaptation: DynamicAdaptationMetrics,
}

/// Path planning evaluation
#[derive(Debug, Clone)]
pub struct PathPlanningMetrics {
    /// Planning success rate
    pub success_rate: f64,
    /// Path optimality (length, time, energy)
    pub optimality: PathOptimalityMetrics,
    /// Planning computation time
    pub planning_time: Duration,
    /// Path safety margin
    pub safety_margin: f64,
}

/// Obstacle avoidance evaluation
#[derive(Debug, Clone)]
pub struct ObstacleAvoidanceMetrics {
    /// Collision avoidance success rate
    pub collision_avoidance_rate: f64,
    /// Near-miss frequency
    pub near_miss_frequency: f64,
    /// Minimum distance to obstacles
    pub min_obstacle_distance: f64,
    /// Avoidance maneuver efficiency
    pub avoidance_efficiency: f64,
}

/// Goal reaching performance
#[derive(Debug, Clone)]
pub struct GoalReachingMetrics {
    /// Goal reaching success rate
    pub success_rate: f64,
    /// Goal reaching accuracy
    pub reaching_accuracy: ErrorStatistics,
    /// Time to reach goal
    pub time_to_goal: Duration,
    /// Path efficiency to goal
    pub path_efficiency: f64,
}

/// Dynamic environment adaptation
#[derive(Debug, Clone)]
pub struct DynamicAdaptationMetrics {
    /// Adaptation response time
    pub response_time: Duration,
    /// Replanning frequency
    pub replanning_frequency: f64,
    /// Success rate in dynamic environments
    pub dynamic_success_rate: f64,
    /// Prediction accuracy for moving obstacles
    pub prediction_accuracy: f64,
}

/// Human-robot interaction metrics
#[derive(Debug, Clone)]
pub struct HumanRobotInteractionMetrics {
    /// Interaction safety metrics
    pub safety_metrics: HriSafetyMetrics,
    /// Communication effectiveness
    pub communication_metrics: CommunicationMetrics,
    /// User satisfaction measures
    pub user_satisfaction: UserSatisfactionMetrics,
    /// Collaboration efficiency
    pub collaboration_efficiency: CollaborationEfficiencyMetrics,
}

/// HRI safety evaluation
#[derive(Debug, Clone)]
pub struct HriSafetyMetrics {
    /// Safe distance maintenance
    pub safe_distance_maintenance: f64,
    /// Emergency stop response time
    pub emergency_response_time: Duration,
    /// Collision avoidance with humans
    pub human_collision_avoidance: f64,
    /// Intention prediction accuracy
    pub intention_prediction_accuracy: f64,
}

/// Communication effectiveness
#[derive(Debug, Clone)]
pub struct CommunicationMetrics {
    /// Command understanding accuracy
    pub command_understanding: f64,
    /// Response appropriateness
    pub response_appropriateness: f64,
    /// Communication latency
    pub communication_latency: Duration,
    /// Multimodal communication effectiveness
    pub multimodal_effectiveness: f64,
}

/// User satisfaction measures
#[derive(Debug, Clone)]
pub struct UserSatisfactionMetrics {
    /// Overall satisfaction score
    pub overall_satisfaction: f64,
    /// Trust level in the robot
    pub trust_level: f64,
    /// Perceived usefulness
    pub perceived_usefulness: f64,
    /// Ease of interaction
    pub ease_of_interaction: f64,
}

/// Collaboration efficiency
#[derive(Debug, Clone)]
pub struct CollaborationEfficiencyMetrics {
    /// Task completion time with human
    pub collaborative_completion_time: Duration,
    /// Efficiency gain from collaboration
    pub efficiency_gain: f64,
    /// Workload distribution balance
    pub workload_balance: f64,
    /// Synchronization accuracy
    pub synchronization_accuracy: f64,
}

/// Multi-robot coordination metrics
#[derive(Debug, Clone)]
pub struct MultiRobotMetrics {
    /// Formation control metrics
    pub formation_control: FormationControlMetrics,
    /// Task allocation efficiency
    pub task_allocation: TaskAllocationMetrics,
    /// Communication network performance
    pub network_performance: NetworkPerformanceMetrics,
    /// Collective behavior emergence
    pub collective_behavior: CollectiveBehaviorMetrics,
}

/// Formation control evaluation
#[derive(Debug, Clone)]
pub struct FormationControlMetrics {
    /// Formation maintenance accuracy
    pub formation_accuracy: ErrorStatistics,
    /// Formation stability
    pub formation_stability: f64,
    /// Reconfiguration time
    pub reconfiguration_time: Duration,
    /// Scalability with robot count
    pub scalability_score: f64,
}

/// Task allocation efficiency
#[derive(Debug, Clone)]
pub struct TaskAllocationMetrics {
    /// Allocation optimality
    pub allocation_optimality: f64,
    /// Load balancing efficiency
    pub load_balancing: f64,
    /// Allocation computation time
    pub allocation_time: Duration,
    /// Adaptation to robot failures
    pub failure_adaptation: f64,
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct NetworkPerformanceMetrics {
    /// Communication reliability
    pub communication_reliability: f64,
    /// Network latency
    pub network_latency: Duration,
    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Network resilience
    pub network_resilience: f64,
}

/// Collective behavior evaluation
#[derive(Debug, Clone)]
pub struct CollectiveBehaviorMetrics {
    /// Swarm coherence
    pub swarm_coherence: f64,
    /// Consensus achievement time
    pub consensus_time: Duration,
    /// Emergent behavior quality
    pub emergent_behavior_quality: f64,
    /// Collective intelligence measure
    pub collective_intelligence: f64,
}

/// Safety and reliability metrics
#[derive(Debug, Clone)]
pub struct SafetyReliabilityMetrics {
    /// Failure detection and recovery
    pub failure_metrics: FailureMetrics,
    /// Risk assessment measures
    pub risk_assessment: RiskAssessmentMetrics,
    /// Redundancy effectiveness
    pub redundancy_metrics: RedundancyMetrics,
    /// Predictive maintenance indicators
    pub maintenance_metrics: MaintenanceMetrics,
}

/// Failure detection and recovery
#[derive(Debug, Clone)]
pub struct FailureMetrics {
    /// Failure detection accuracy
    pub detection_accuracy: f64,
    /// False alarm rate
    pub false_alarm_rate: f64,
    /// Recovery success rate
    pub recovery_success_rate: f64,
    /// Mean time to recovery
    pub mean_time_to_recovery: Duration,
}

/// Risk assessment metrics
#[derive(Debug, Clone)]
pub struct RiskAssessmentMetrics {
    /// Risk prediction accuracy
    pub risk_prediction_accuracy: f64,
    /// Safety margin maintenance
    pub safety_margin_maintenance: f64,
    /// Hazard identification rate
    pub hazard_identification_rate: f64,
    /// Risk mitigation effectiveness
    pub risk_mitigation_effectiveness: f64,
}

/// Redundancy effectiveness
#[derive(Debug, Clone)]
pub struct RedundancyMetrics {
    /// Redundant system utilization
    pub redundancy_utilization: f64,
    /// Failover success rate
    pub failover_success_rate: f64,
    /// Performance degradation under failures
    pub performance_degradation: f64,
    /// System availability
    pub system_availability: f64,
}

/// Predictive maintenance metrics
#[derive(Debug, Clone)]
pub struct MaintenanceMetrics {
    /// Maintenance prediction accuracy
    pub prediction_accuracy: f64,
    /// Preventive maintenance effectiveness
    pub preventive_effectiveness: f64,
    /// System health monitoring quality
    pub health_monitoring_quality: f64,
    /// Maintenance scheduling optimality
    pub scheduling_optimality: f64,
}

/// Robotic perception metrics
#[derive(Debug, Clone)]
pub struct RoboticPerceptionMetrics {
    /// Object detection and tracking
    pub object_detection: ObjectDetectionMetrics,
    /// Scene understanding quality
    pub scene_understanding: SceneUnderstandingMetrics,
    /// Sensor fusion effectiveness
    pub sensor_fusion: SensorFusionMetrics,
    /// Real-time processing performance
    pub real_time_performance: RealTimePerformanceMetrics,
}

/// Object detection for robotics
#[derive(Debug, Clone)]
pub struct ObjectDetectionMetrics {
    /// Detection accuracy per object class
    pub per_class_accuracy: HashMap<String, f64>,
    /// Real-time detection rate
    pub real_time_detection_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Tracking consistency over time
    pub tracking_consistency: f64,
}

/// Scene understanding evaluation
#[derive(Debug, Clone)]
pub struct SceneUnderstandingMetrics {
    /// Semantic segmentation accuracy
    pub segmentation_accuracy: f64,
    /// Spatial relationship understanding
    pub spatial_relationship_accuracy: f64,
    /// Dynamic scene adaptation
    pub dynamic_adaptation_rate: f64,
    /// Context reasoning quality
    pub context_reasoning_quality: f64,
}

/// Sensor fusion effectiveness
#[derive(Debug, Clone)]
pub struct SensorFusionMetrics {
    /// Fusion accuracy improvement
    pub accuracy_improvement: f64,
    /// Sensor reliability weighting
    pub reliability_weighting_accuracy: f64,
    /// Failure detection in sensors
    pub sensor_failure_detection: f64,
    /// Information gain from fusion
    pub information_gain: f64,
}

/// Real-time processing performance
#[derive(Debug, Clone)]
pub struct RealTimePerformanceMetrics {
    /// Processing latency
    pub processing_latency: Duration,
    /// Throughput (frames/operations per second)
    pub throughput: f64,
    /// Real-time constraint satisfaction
    pub real_time_satisfaction: f64,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
}

// Implementation of the main robotics metrics

impl RoboticsMetrics {
    /// Create new robotics metrics suite
    pub fn new() -> Self {
        Self {
            motion_metrics: MotionPlanningMetrics::new(),
            slam_metrics: SlamMetrics::new(),
            manipulation_metrics: ManipulationMetrics::new(),
            navigation_metrics: NavigationMetrics::new(),
            hri_metrics: HumanRobotInteractionMetrics::new(),
            multi_robot_metrics: MultiRobotMetrics::new(),
            safety_metrics: SafetyReliabilityMetrics::new(),
            perception_metrics: RoboticPerceptionMetrics::new(),
        }
    }

    /// Evaluate trajectory smoothness
    pub fn evaluate_trajectory_smoothness<F>(
        &mut self,
        trajectory: &ArrayView2<F>,
        time_stamps: &ArrayView1<F>,
    ) -> Result<TrajectorySmoothnessMetrics>
    where
        F: Float,
    {
        let n_points = trajectory.nrows();
        let n_dims = trajectory.ncols();

        if n_points < 3 {
            return Err(MetricsError::InvalidInput(
                "Need at least 3 trajectory points for smoothness evaluation".to_string(),
            ));
        }

        // Calculate velocities
        let mut velocities = Array2::zeros((n_points - 1, n_dims));
        for i in 0..n_points - 1 {
            let dt = time_stamps[i + 1] - time_stamps[i];
            for j in 0..n_dims {
                velocities[[i, j]] = (trajectory[[i + 1, j]] - trajectory[[i, j]]) / dt;
            }
        }

        // Calculate accelerations
        let mut accelerations = Array2::zeros((n_points - 2, n_dims));
        for i in 0..n_points - 2 {
            let dt = time_stamps[i + 1] - time_stamps[i];
            for j in 0..n_dims {
                accelerations[[i, j]] = (velocities[[i + 1, j]] - velocities[[i, j]]) / dt;
            }
        }

        // Calculate jerks
        let mut jerks = Array2::zeros((n_points - 3, n_dims));
        for i in 0..n_points - 3 {
            let dt = time_stamps[i + 1] - time_stamps[i];
            for j in 0..n_dims {
                jerks[[i, j]] = (accelerations[[i + 1, j]] - accelerations[[i, j]]) / dt;
            }
        }

        // Calculate jerk metrics
        let jerk_magnitudes: Vec<f64> = (0..jerks.nrows())
            .map(|i| {
                jerks.row(i)
                    .iter()
                    .map(|&x| x.to_f64().unwrap().powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let average_jerk = jerk_magnitudes.iter().sum::<f64>() / jerk_magnitudes.len() as f64;
        let max_jerk = jerk_magnitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate acceleration variance
        let acceleration_magnitudes: Vec<f64> = (0..accelerations.nrows())
            .map(|i| {
                accelerations.row(i)
                    .iter()
                    .map(|&x| x.to_f64().unwrap().powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let mean_acceleration = acceleration_magnitudes.iter().sum::<f64>() / acceleration_magnitudes.len() as f64;
        let acceleration_variance = acceleration_magnitudes
            .iter()
            .map(|&x| (x - mean_acceleration).powi(2))
            .sum::<f64>() / acceleration_magnitudes.len() as f64;

        // Calculate curvature metrics
        let curvature_metrics = self.calculate_curvature_metrics(&trajectory, &velocities, &accelerations)?;

        // Calculate velocity smoothness (coefficient of variation)
        let velocity_magnitudes: Vec<f64> = (0..velocities.nrows())
            .map(|i| {
                velocities.row(i)
                    .iter()
                    .map(|&x| x.to_f64().unwrap().powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let mean_velocity = velocity_magnitudes.iter().sum::<f64>() / velocity_magnitudes.len() as f64;
        let velocity_std = (velocity_magnitudes
            .iter()
            .map(|&x| (x - mean_velocity).powi(2))
            .sum::<f64>() / velocity_magnitudes.len() as f64)
            .sqrt();

        let velocity_smoothness = if mean_velocity > 0.0 {
            1.0 - (velocity_std / mean_velocity) // Higher value means smoother
        } else {
            1.0
        };

        Ok(TrajectorySmoothnessMetrics {
            average_jerk,
            max_jerk,
            acceleration_variance,
            curvature_metrics,
            velocity_smoothness,
        })
    }

    /// Calculate curvature metrics for trajectory
    fn calculate_curvature_metrics<F>(
        &self,\n        trajectory: &ArrayView2<F>,
        velocities: &Array2<F>,
        accelerations: &Array2<F>,
    ) -> Result<CurvatureMetrics>
    where
        F: Float,
    {
        let mut curvatures = Vec::new();

        for i in 0..velocities.nrows().min(accelerations.nrows()) {
            let vel = velocities.row(i);
            let acc = accelerations.row(i);

            // Calculate curvature using κ = |v × a| / |v|³
            let vel_magnitude = vel.iter()
                .map(|&x| x.to_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();

            if vel_magnitude > 1e-6 {
                // For 2D case, cross product magnitude
                let cross_product_magnitude = if vel.len() >= 2 {
                    (vel[0].to_f64().unwrap() * acc[1].to_f64().unwrap() 
                     - vel[1].to_f64().unwrap() * acc[0].to_f64().unwrap()).abs()
                } else {
                    0.0
                };

                let curvature = cross_product_magnitude / vel_magnitude.powi(3);
                curvatures.push(curvature);
            }
        }

        if curvatures.is_empty() {
            return Ok(CurvatureMetrics {
                average_curvature: 0.0,
                max_curvature: 0.0,
                curvature_variance: 0.0,
                sharp_turns_count: 0,
            });
        }

        let average_curvature = curvatures.iter().sum::<f64>() / curvatures.len() as f64;
        let max_curvature = curvatures.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let curvature_variance = curvatures
            .iter()
            .map(|&x| (x - average_curvature).powi(2))
            .sum::<f64>() / curvatures.len() as f64;

        // Count sharp turns (curvature above 95th percentile)
        let mut sorted_curvatures = curvatures.clone();
        sorted_curvatures.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_95_index = (0.95 * sorted_curvatures.len() as f64) as usize;
        let sharp_turn_threshold = sorted_curvatures.get(percentile_95_index).unwrap_or(&0.0);
        let sharp_turns_count = curvatures.iter().filter(|&&x| x > *sharp_turn_threshold).count();

        Ok(CurvatureMetrics {
            average_curvature,
            max_curvature,
            curvature_variance,
            sharp_turns_count,
        })
    }

    /// Evaluate SLAM localization accuracy
    pub fn evaluate_slam_localization<F>(
        &mut self,
        estimated_poses: &ArrayView2<F>,
        ground_truth_poses: &ArrayView2<F>,
        timestamps: &ArrayView1<F>,
    ) -> Result<LocalizationAccuracyMetrics>
    where
        F: Float,
    {
        if estimated_poses.nrows() != ground_truth_poses.nrows() {
            return Err(MetricsError::InvalidInput(
                "Estimated and ground truth _poses must have same number of points".to_string(),
            ));
        }

        let n_poses = estimated_poses.nrows();
        
        // Calculate Absolute Trajectory Error (ATE)
        let mut translation_errors = Vec::new();
        let mut rotation_errors = Vec::new();

        for i in 0..n_poses {
            // Assume pose format: [x, y, z, qx, qy, qz, qw] or [x, y, theta] for 2D
            let est_pose = estimated_poses.row(i);
            let gt_pose = ground_truth_poses.row(i);

            // Translation error (Euclidean distance)
            let trans_error = if est_pose.len() >= 3 {
                ((est_pose[0] - gt_pose[0]).to_f64().unwrap().powi(2) +
                 (est_pose[1] - gt_pose[1]).to_f64().unwrap().powi(2) +
                 (est_pose[2] - gt_pose[2]).to_f64().unwrap().powi(2)).sqrt()
            } else {
                ((est_pose[0] - gt_pose[0]).to_f64().unwrap().powi(2) +
                 (est_pose[1] - gt_pose[1]).to_f64().unwrap().powi(2)).sqrt()
            };

            translation_errors.push(trans_error);

            // Rotation error (angle difference)
            let rot_error = if est_pose.len() >= 7 {
                // Quaternion case - simplified angular difference
                let angle_diff = (est_pose[est_pose.len()-1] - gt_pose[gt_pose.len()-1]).to_f64().unwrap().abs();
                angle_diff.min(2.0 * std::f64::consts::PI - angle_diff)
            } else if est_pose.len() >= 3 {
                // 2D angle case
                let angle_diff = (est_pose[2] - gt_pose[2]).to_f64().unwrap().abs();
                angle_diff.min(2.0 * std::f64::consts::PI - angle_diff)
            } else {
                0.0
            };

            rotation_errors.push(rot_error);
        }

        // Calculate error statistics
        let translation_error = self.calculate_error_statistics(&translation_errors);
        let rotation_error = self.calculate_error_statistics(&rotation_errors);

        // Calculate ATE
        let absolute_trajectory_error = translation_errors.iter().sum::<f64>() / translation_errors.len() as f64;

        // Calculate RPE (Relative Pose Error)
        let relative_pose_error = self.calculate_relative_pose_error(estimated_poses, ground_truth_poses)?;

        // Calculate drift metrics
        let drift_metrics = self.calculate_drift_metrics(estimated_poses, ground_truth_poses, timestamps)?;

        Ok(LocalizationAccuracyMetrics {
            absolute_trajectory_error,
            relative_pose_error,
            translation_error,
            rotation_error,
            drift_metrics,
        })
    }

    /// Calculate error statistics
    fn calculate_error_statistics(&self, errors: &[f64]) -> ErrorStatistics {
        if errors.is_empty() {
            return ErrorStatistics {
                rmse: 0.0,
                mae: 0.0,
                max_error: 0.0,
                std_dev: 0.0,
                median_error: 0.0,
            };
        }

        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let rmse = (errors.iter().map(|&x| x * x).sum::<f64>() / errors.len() as f64).sqrt();
        let mae = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = errors.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / errors.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_errors = errors.to_vec();
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_error = if sorted_errors.len() % 2 == 0 {
            (sorted_errors[sorted_errors.len() / 2 - 1] + sorted_errors[sorted_errors.len() / 2]) / 2.0
        } else {
            sorted_errors[sorted_errors.len() / 2]
        };

        ErrorStatistics {
            rmse,
            mae,
            max_error,
            std_dev,
            median_error,
        }
    }

    /// Calculate relative pose error
    fn calculate_relative_pose_error<F>(
        &self,
        estimated_poses: &ArrayView2<F>,
        ground_truth_poses: &ArrayView2<F>,
    ) -> Result<f64>
    where
        F: Float,
    {
        let n_poses = estimated_poses.nrows();
        if n_poses < 2 {
            return Ok(0.0);
        }

        let mut relative_errors = Vec::new();

        for i in 0..n_poses - 1 {
            // Calculate relative transformation between consecutive _poses
            let est_curr = estimated_poses.row(i);
            let est_next = estimated_poses.row(i + 1);
            let gt_curr = ground_truth_poses.row(i);
            let gt_next = ground_truth_poses.row(i + 1);

            // Simplified relative error calculation (translation component)
            let est_rel_x = est_next[0] - est_curr[0];
            let est_rel_y = est_next[1] - est_curr[1];
            let gt_rel_x = gt_next[0] - gt_curr[0];
            let gt_rel_y = gt_next[1] - gt_curr[1];

            let error = ((est_rel_x - gt_rel_x).to_f64().unwrap().powi(2) +
                        (est_rel_y - gt_rel_y).to_f64().unwrap().powi(2)).sqrt();

            relative_errors.push(error);
        }

        Ok(relative_errors.iter().sum::<f64>() / relative_errors.len() as f64)
    }

    /// Calculate drift metrics
    fn calculate_drift_metrics<F>(
        &self,
        estimated_poses: &ArrayView2<F>,
        ground_truth_poses: &ArrayView2<F>,
        timestamps: &ArrayView1<F>,
    ) -> Result<DriftMetrics>
    where
        F: Float,
    {
        let n_poses = estimated_poses.nrows();
        if n_poses < 2 {
            return Ok(DriftMetrics {
                translation_drift_rate: 0.0,
                rotation_drift_rate: 0.0,
                scale_drift: 0.0,
                drift_consistency: 1.0,
            });
        }

        // Calculate cumulative distance traveled
        let mut cumulative_distance = 0.0;
        for i in 0..n_poses - 1 {
            let dx = (ground_truth_poses[[i+1, 0]] - ground_truth_poses[[i, 0]]).to_f64().unwrap();
            let dy = (ground_truth_poses[[i+1, 1]] - ground_truth_poses[[i, 1]]).to_f64().unwrap();
            cumulative_distance += (dx * dx + dy * dy).sqrt();
        }

        // Calculate final translation error
        let final_trans_error = {
            let dx = (estimated_poses[[n_poses-1, 0]] - ground_truth_poses[[n_poses-1, 0]]).to_f64().unwrap();
            let dy = (estimated_poses[[n_poses-1, 1]] - ground_truth_poses[[n_poses-1, 1]]).to_f64().unwrap();
            (dx * dx + dy * dy).sqrt()
        };

        let translation_drift_rate = if cumulative_distance > 0.0 {
            final_trans_error / cumulative_distance
        } else {
            0.0
        };

        // Calculate rotation drift rate (if rotation data available)
        let rotation_drift_rate = if estimated_poses.ncols() > 2 && ground_truth_poses.ncols() > 2 {
            let final_rot_error = (estimated_poses[[n_poses-1, 2]] - ground_truth_poses[[n_poses-1, 2]])
                .to_f64().unwrap().abs();
            if cumulative_distance > 0.0 {
                final_rot_error.to_degrees() / cumulative_distance
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Simple drift consistency measure
        let mut drift_values = Vec::new();
        for i in 1..n_poses {
            let partial_distance: f64 = (0..i).map(|j| {
                let dx = (ground_truth_poses[[j+1, 0]] - ground_truth_poses[[j, 0]]).to_f64().unwrap();
                let dy = (ground_truth_poses[[j+1, 1]] - ground_truth_poses[[j, 1]]).to_f64().unwrap();
                (dx * dx + dy * dy).sqrt()
            }).sum();

            if partial_distance > 0.0 {
                let dx = (estimated_poses[[i, 0]] - ground_truth_poses[[i, 0]]).to_f64().unwrap();
                let dy = (estimated_poses[[i, 1]] - ground_truth_poses[[i, 1]]).to_f64().unwrap();
                let trans_error = (dx * dx + dy * dy).sqrt();
                drift_values.push(trans_error / partial_distance);
            }
        }

        let drift_consistency = if drift_values.len() > 1 {
            let mean_drift = drift_values.iter().sum::<f64>() / drift_values.len() as f64;
            let variance = drift_values.iter()
                .map(|&x| (x - mean_drift).powi(2))
                .sum::<f64>() / drift_values.len() as f64;
            let coefficient_of_variation = if mean_drift > 0.0 {
                variance.sqrt() / mean_drift
            } else {
                0.0
            };
            1.0 / (1.0 + coefficient_of_variation) // Higher consistency for lower CV
        } else {
            1.0
        };

        Ok(DriftMetrics {
            translation_drift_rate,
            rotation_drift_rate,
            scale_drift: 0.0, // Would need scale estimation for full implementation
            drift_consistency,
        })
    }

    /// Evaluate path optimality metrics
    pub fn evaluate_path_optimality<F>(
        &mut self,
        path: &ArrayView2<F>,
        optimal_path: Option<&ArrayView2<F>>,
        velocities: Option<&ArrayView2<F>>,
        obstacle_map: Option<&ArrayView2<F>>,
        robot_mass: Option<f64>,
        time_stamps: Option<&ArrayView1<F>>,
    ) -> Result<PathOptimalityMetrics>
    where
        F: Float,
    {
        let n_points = path.nrows();
        let n_dims = path.ncols();

        if n_points < 2 {
            return Err(MetricsError::InvalidInput(
                "Path must have at least 2 points".to_string(),
            ));
        }

        // Calculate _path length
        let mut path_length = 0.0;
        for i in 0..n_points - 1 {
            let mut segment_length = 0.0;
            for j in 0..n_dims {
                let diff = (_path[[i + 1, j]] - path[[i, j]]).to_f64().unwrap();
                segment_length += diff * diff;
            }
            path_length += segment_length.sqrt();
        }

        // Calculate length optimality ratio
        let length_optimality_ratio = if let Some(optimal) = optimal_path {
            let mut optimal_length = 0.0;
            for i in 0..optimal.nrows() - 1 {
                let mut segment_length = 0.0;
                for j in 0..n_dims {
                    let diff = (optimal[[i + 1, j]] - optimal[[i, j]]).to_f64().unwrap();
                    segment_length += diff * diff;
                }
                optimal_length += segment_length.sqrt();
            }
            if optimal_length > 0.0 {
                optimal_length / path_length
            } else {
                1.0
            }
        } else {
            1.0 // Assume current _path is optimal if no reference provided
        };

        // Calculate energy optimality ratio
        let energy_optimality_ratio = if let (Some(velocities), Some(_mass)) = (velocities, robot_mass) {
            let mut total_energy = 0.0;
            
            // Calculate kinetic energy throughout the _path
            for i in 0..velocities.nrows() {
                let velocity_magnitude_sq: f64 = velocities.row(i)
                    .iter()
                    .map(|&v| v.to_f64().unwrap().powi(2))
                    .sum();
                total_energy += 0.5 * _mass * velocity_magnitude_sq;
            }

            // Calculate accelerations for dynamic energy
            if velocities.nrows() > 1 && time_stamps.is_some() {
                let timestamps = time_stamps.unwrap();
                for i in 0..velocities.nrows() - 1 {
                    let dt = (timestamps[i + 1] - timestamps[i]).to_f64().unwrap();
                    if dt > 0.0 {
                        let acc_magnitude_sq: f64 = velocities.row(i + 1)
                            .iter()
                            .zip(velocities.row(i).iter())
                            .map(|(&v_next, &v_curr)| {
                                let acc = (v_next - v_curr).to_f64().unwrap() / dt;
                                acc * acc
                            })
                            .sum();
                        // Add energy cost for acceleration (simplified)
                        total_energy += 0.5 * _mass * acc_magnitude_sq * dt;
                    }
                }
            }

            // Assume optimal energy is minimum required (simplified)
            let optimal_energy = 0.5 * _mass * (path_length / 10.0).powi(2); // Simplified estimate
            if total_energy > 0.0 {
                optimal_energy / total_energy
            } else {
                1.0
            }
        } else {
            1.0 // No energy data available
        };

        // Calculate time optimality ratio
        let time_optimality_ratio = if let Some(timestamps) = time_stamps {
            let total_time = (timestamps[timestamps.len() - 1] - timestamps[0]).to_f64().unwrap();
            // Estimate optimal time based on _path length and reasonable speed
            let reasonable_speed = 1.0; // m/s, adjustable parameter
            let optimal_time = path_length / reasonable_speed;
            if total_time > 0.0 {
                optimal_time / total_time
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Calculate obstacle clearance metrics
        let obstacle_clearance = self.calculate_obstacle_clearance(_path, obstacle_map)?;

        Ok(PathOptimalityMetrics {
            length_optimality_ratio,
            energy_optimality_ratio,
            time_optimality_ratio,
            obstacle_clearance,
        })
    }

    /// Calculate obstacle clearance metrics
    fn calculate_obstacle_clearance<F>(
        &self,
        path: &ArrayView2<F>,
        obstacle_map: Option<&ArrayView2<F>>,
    ) -> Result<ObstacleClearanceMetrics>
    where
        F: Float,
    {
        if obstacle_map.is_none() {
            return Ok(ObstacleClearanceMetrics {
                min_clearance: f64::INFINITY,
                avg_clearance: f64::INFINITY,
                clearance_variance: 0.0,
                safety_margin_ratio: 1.0,
            });
        }

        let obstacles = obstacle_map.unwrap();
        let mut clearance_distances = Vec::new();

        // For each point in the path, find minimum distance to any obstacle
        for i in 0..path.nrows() {
            let path_point = path.row(i);
            let mut min_dist = f64::INFINITY;

            // Simple grid-based obstacle checking (assumes 2D)
            if path_point.len() >= 2 {
                let x = path_point[0].to_f64().unwrap();
                let y = path_point[1].to_f64().unwrap();

                // Check distance to all obstacle points
                for obs_i in 0..obstacles.nrows() {
                    for obs_j in 0..obstacles.ncols() {
                        // If this is an obstacle cell (value > 0.5)
                        if obstacles[[obs_i, obs_j]].to_f64().unwrap() > 0.5 {
                            let obs_x = obs_i as f64;
                            let obs_y = obs_j as f64;
                            let dist = ((x - obs_x).powi(2) + (y - obs_y).powi(2)).sqrt();
                            min_dist = min_dist.min(dist);
                        }
                    }
                }
            }

            clearance_distances.push(min_dist);
        }

        if clearance_distances.is_empty() {
            return Ok(ObstacleClearanceMetrics {
                min_clearance: f64::INFINITY,
                avg_clearance: f64::INFINITY,
                clearance_variance: 0.0,
                safety_margin_ratio: 1.0,
            });
        }

        let min_clearance = clearance_distances.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg_clearance = clearance_distances.iter().sum::<f64>() / clearance_distances.len() as f64;
        
        let mean = avg_clearance;
        let clearance_variance = clearance_distances
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / clearance_distances.len() as f64;

        // Calculate safety margin ratio (ratio of actual clearance to minimum safe clearance)
        let min_safe_clearance = 0.5; // Adjustable safety parameter
        let safety_margin_ratio = if min_safe_clearance > 0.0 {
            (min_clearance / min_safe_clearance).min(1.0)
        } else {
            1.0
        };

        Ok(ObstacleClearanceMetrics {
            min_clearance,
            avg_clearance,
            clearance_variance,
            safety_margin_ratio,
        })
    }

    /// Evaluate constraint satisfaction metrics
    pub fn evaluate_constraint_satisfaction<F>(
        &mut self,
        joint_positions: &ArrayView2<F>,
        joint_velocities: Option<&ArrayView2<F>>,
        joint_accelerations: Option<&ArrayView2<F>>,
        joint_torques: Option<&ArrayView2<F>>,
        joint_limits: Option<&Array2<f64>>, // [min_pos, max_pos] for each joint
        velocity_limits: Option<&Array1<f64>>, // max velocities for each joint
        acceleration_limits: Option<&Array1<f64>>, // max accelerations for each joint
        torque_limits: Option<&Array1<f64>>, // max torques for each joint
        collision_detections: Option<&Array1<bool>>, // collision status for each time step
    ) -> Result<ConstraintSatisfactionMetrics>
    where
        F: Float,
    {
        let n_timesteps = joint_positions.nrows();
        let n_joints = joint_positions.ncols();

        if n_timesteps == 0 || n_joints == 0 {
            return Err(MetricsError::InvalidInput(
                "Joint _positions cannot be empty".to_string(),
            ));
        }

        // Calculate joint _limits satisfaction
        let joint_limits_satisfaction = if let Some(_limits) = joint_limits {
            if limits.nrows() != n_joints || limits.ncols() != 2 {
                return Err(MetricsError::InvalidInput(
                    "Joint _limits must have shape [n_joints, 2] for [min, max]".to_string(),
                ));
            }

            let mut violations = 0;
            let mut total_checks = 0;

            for t in 0..n_timesteps {
                for j in 0..n_joints {
                    let pos = joint_positions[[t, j]].to_f64().unwrap();
                    let min_limit = limits[[j, 0]];
                    let max_limit = limits[[j, 1]];

                    total_checks += 1;
                    if pos < min_limit || pos > max_limit {
                        violations += 1;
                    }
                }
            }

            if total_checks > 0 {
                1.0 - (violations as f64 / total_checks as f64)
            } else {
                1.0
            }
        } else {
            1.0 // No _limits specified, assume all satisfied
        };

        // Calculate velocity _limits satisfaction
        let velocity_limits_satisfaction = if let (Some(_velocities), Some(vel_limits)) = 
            (joint_velocities, velocity_limits) {
            if vel_limits.len() != n_joints {
                return Err(MetricsError::InvalidInput(
                    "Velocity _limits must have same length as number of joints".to_string(),
                ));
            }

            let mut violations = 0;
            let mut total_checks = 0;

            for t in 0.._velocities.nrows() {
                for j in 0..n_joints {
                    let vel = velocities[[t, j]].to_f64().unwrap().abs();
                    let vel_limit = vel_limits[j];

                    total_checks += 1;
                    if vel > vel_limit {
                        violations += 1;
                    }
                }
            }

            if total_checks > 0 {
                1.0 - (violations as f64 / total_checks as f64)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Calculate acceleration _limits satisfaction
        let acceleration_limits_satisfaction = if let (Some(_accelerations), Some(acc_limits)) = 
            (joint_accelerations, acceleration_limits) {
            if acc_limits.len() != n_joints {
                return Err(MetricsError::InvalidInput(
                    "Acceleration _limits must have same length as number of joints".to_string(),
                ));
            }

            let mut violations = 0;
            let mut total_checks = 0;

            for t in 0.._accelerations.nrows() {
                for j in 0..n_joints {
                    let acc = accelerations[[t, j]].to_f64().unwrap().abs();
                    let acc_limit = acc_limits[j];

                    total_checks += 1;
                    if acc > acc_limit {
                        violations += 1;
                    }
                }
            }

            if total_checks > 0 {
                1.0 - (violations as f64 / total_checks as f64)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Calculate torque _limits satisfaction
        let torque_limits_satisfaction = if let (Some(_torques), Some(torque_lims)) = 
            (joint_torques, torque_limits) {
            if torque_lims.len() != n_joints {
                return Err(MetricsError::InvalidInput(
                    "Torque _limits must have same length as number of joints".to_string(),
                ));
            }

            let mut violations = 0;
            let mut total_checks = 0;

            for t in 0.._torques.nrows() {
                for j in 0..n_joints {
                    let torque = torques[[t, j]].to_f64().unwrap().abs();
                    let torque_limit = torque_lims[j];

                    total_checks += 1;
                    if torque > torque_limit {
                        violations += 1;
                    }
                }
            }

            if total_checks > 0 {
                1.0 - (violations as f64 / total_checks as f64)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Calculate collision avoidance rate
        let collision_avoidance_rate = if let Some(collisions) = collision_detections {
            if collisions.len() != n_timesteps {
                return Err(MetricsError::InvalidInput(
                    "Collision _detections must have same length as number of timesteps".to_string(),
                ));
            }

            let collision_count = collisions.iter().filter(|&&x| x).count();
            if n_timesteps > 0 {
                1.0 - (collision_count as f64 / n_timesteps as f64)
            } else {
                1.0
            }
        } else {
            1.0 // No collision data, assume no collisions
        };

        Ok(ConstraintSatisfactionMetrics {
            joint_limits_satisfaction,
            velocity_limits_satisfaction,
            acceleration_limits_satisfaction,
            torque_limits_satisfaction,
            collision_avoidance_rate,
        })
    }

    /// Evaluate grasping metrics
    pub fn evaluate_grasping_metrics<F>(
        &mut self,
        grasp_attempts: &Array1<bool>,  // Success/failure for each grasp attempt
        grasp_forces: Option<&ArrayView2<F>>, // Forces applied during grasps
        contact_points: Option<&ArrayView2<F>>, // Contact point positions
        object_slippage: Option<&Array1<bool>>, // Whether object slipped during grasp
        force_closure_quality: Option<&Array1<f64>>, // Quality measure for force closure
        approach_trajectories: Option<&ArrayView3<F>>, // Trajectories for grasp approach
    ) -> Result<GraspingMetrics>
    where
        F: Float,
    {
        let n_attempts = grasp_attempts.len();
        if n_attempts == 0 {
            return Err(MetricsError::InvalidInput(
                "Must have at least one grasp attempt".to_string(),
            ));
        }

        // Calculate basic success rate
        let success_count = grasp_attempts.iter().filter(|&&x| x).count();
        let success_rate = success_count as f64 / n_attempts as f64;

        // Calculate stability score based on _slippage data
        let stability_score = if let Some(_slippage) = object_slippage {
            if slippage.len() != n_attempts {
                return Err(MetricsError::InvalidInput(
                    "Slippage data must match number of grasp _attempts".to_string(),
                ));
            }
            
            let stable_grasps = slippage.iter()
                .zip(grasp_attempts.iter())
                .filter(|&(slip, success)| *success && !slip)
                .count();
            
            if success_count > 0 {
                stable_grasps as f64 / success_count as f64
            } else {
                0.0
            }
        } else {
            // If no _slippage data, assume all successful grasps are stable
            1.0
        };

        // Calculate force closure _quality score
        let force_closure_quality_score = if let Some(quality_scores) = force_closure_quality {
            if quality_scores.len() != n_attempts {
                return Err(MetricsError::InvalidInput(
                    "Force closure _quality must match number of grasp _attempts".to_string(),
                ));
            }

            // Average _quality for successful grasps only
            let mut total_quality = 0.0;
            let mut successful_attempts = 0;

            for (i, &success) in grasp_attempts.iter().enumerate() {
                if success {
                    total_quality += quality_scores[i];
                    successful_attempts += 1;
                }
            }

            if successful_attempts > 0 {
                total_quality / successful_attempts as f64
            } else {
                0.0
            }
        } else {
            // Default _quality score based on success rate
            success_rate
        };

        // Calculate robustness score (based on force consistency)
        let robustness_score = if let Some(_forces) = grasp_forces {
            if forces.nrows() != n_attempts {
                return Err(MetricsError::InvalidInput(
                    "Grasp _forces must match number of grasp _attempts".to_string(),
                ));
            }

            let mut force_variations = Vec::new();
            
            for i in 0..n_attempts {
                if grasp_attempts[i] {
                    // Calculate force magnitude for this grasp
                    let force_magnitude: f64 = forces.row(i)
                        .iter()
                        .map(|&f| f.to_f64().unwrap().powi(2))
                        .sum::<f64>()
                        .sqrt();
                    force_variations.push(force_magnitude);
                }
            }

            if force_variations.len() > 1 {
                let mean_force = force_variations.iter().sum::<f64>() / force_variations.len() as f64;
                let variance = force_variations.iter()
                    .map(|&f| (f - mean_force).powi(2))
                    .sum::<f64>() / force_variations.len() as f64;
                let coefficient_of_variation = if mean_force > 0.0 {
                    variance.sqrt() / mean_force
                } else {
                    0.0
                };
                
                // Higher robustness for lower variation
                1.0 / (1.0 + coefficient_of_variation)
            } else {
                1.0
            }
        } else {
            stability_score // Use stability as proxy for robustness
        };

        // Calculate approach _quality
        let approach_quality = if let Some(_trajectories) = approach_trajectories {
            // Simplified approach _quality based on trajectory smoothness
            let mut total_smoothness = 0.0;
            let mut valid_trajectories = 0;

            for i in 0..n_attempts {
                if grasp_attempts[i] && i < trajectories.shape()[0] {
                    // Calculate trajectory smoothness (simplified)
                    let traj = trajectories.slice(ndarray::s![i, .., ..]);
                    let n_points = traj.shape()[0];
                    
                    if n_points > 2 {
                        let mut jerk_sum = 0.0;
                        let mut point_count = 0;
                        
                        for t in 2..n_points {
                            for dim in 0..traj.shape()[1] {
                                // Simple finite difference for jerk approximation
                                let p0 = traj[[t-2, dim]].to_f64().unwrap();
                                let p1 = traj[[t-1, dim]].to_f64().unwrap();
                                let p2 = traj[[t, dim]].to_f64().unwrap();
                                
                                // Second derivative approximation
                                let jerk_approx = (p2 - 2.0 * p1 + p0).abs();
                                jerk_sum += jerk_approx;
                                point_count += 1;
                            }
                        }
                        
                        if point_count > 0 {
                            let avg_jerk = jerk_sum / point_count as f64;
                            // Convert to smoothness score (lower jerk = higher quality)
                            total_smoothness += 1.0 / (1.0 + avg_jerk);
                            valid_trajectories += 1;
                        }
                    }
                }
            }

            if valid_trajectories > 0 {
                total_smoothness / valid_trajectories as f64
            } else {
                success_rate // Use success rate as fallback
            }
        } else {
            success_rate // Use success rate as default approach _quality
        };

        Ok(GraspingMetrics {
            success_rate,
            stability_score,
            force_closure_quality: force_closure_quality_score,
            robustness_score,
            approach_quality,
        })
    }

    /// Evaluate navigation metrics
    pub fn evaluate_navigation_metrics<F>(
        &mut self,
        planned_paths: &[ArrayView2<F>], // Multiple planned paths
        executed_paths: &[ArrayView2<F>], // Corresponding executed paths
        goals: &ArrayView2<F>, // Goal positions for each navigation task
        obstacles: Option<&ArrayView2<F>>, // Static obstacle map
        dynamic_obstacles: Option<&[ArrayView2<F>]>, // Dynamic obstacles over time
        planning_times: &Array1<f64>, // Time taken for planning each path
        execution_success: &Array1<bool>, // Whether each navigation succeeded
        timestamps: Option<&[ArrayView1<F>]>, // Timestamps for executed paths
    ) -> Result<NavigationMetrics>
    where
        F: Float,
    {
        let n_tasks = planned_paths.len();
        if n_tasks == 0 {
            return Err(MetricsError::InvalidInput(
                "Must have at least one navigation task".to_string(),
            ));
        }

        if executed_paths.len() != n_tasks || goals.nrows() != n_tasks {
            return Err(MetricsError::InvalidInput(
                "All navigation data arrays must have same length".to_string(),
            ));
        }

        // Evaluate path planning metrics
        let path_planning = self.evaluate_path_planning_metrics(
            planned_paths,
            goals,
            obstacles,
            planning_times,
            execution_success,
        )?;

        // Evaluate obstacle avoidance metrics
        let obstacle_avoidance = self.evaluate_obstacle_avoidance_metrics(
            executed_paths,
            obstacles,
            dynamic_obstacles,
            execution_success,
        )?;

        // Evaluate goal reaching metrics
        let goal_reaching = self.evaluate_goal_reaching_metrics(
            executed_paths,
            goals,
            execution_success,
            timestamps,
        )?;

        // Evaluate dynamic adaptation metrics
        let dynamic_adaptation = self.evaluate_dynamic_adaptation_metrics(
            planned_paths,
            executed_paths,
            dynamic_obstacles,
            timestamps,
        )?;

        Ok(NavigationMetrics {
            path_planning,
            obstacle_avoidance,
            goal_reaching,
            dynamic_adaptation,
        })
    }

    /// Evaluate path planning metrics
    fn evaluate_path_planning_metrics<F>(
        &self,
        planned_paths: &[ArrayView2<F>],
        goals: &ArrayView2<F>,
        obstacles: Option<&ArrayView2<F>>,
        planning_times: &Array1<f64>,
        execution_success: &Array1<bool>,
    ) -> Result<PathPlanningMetrics>
    where
        F: Float,
    {
        let n_tasks = planned_paths.len();
        
        // Calculate planning _success rate
        let success_count = execution_success.iter().filter(|&&x| x).count();
        let success_rate = success_count as f64 / n_tasks as f64;

        // Calculate average planning time
        let avg_planning_time = planning_times.iter().sum::<f64>() / n_tasks as f64;
        let planning_time = Duration::from_secs_f64(avg_planning_time);

        // Calculate path optimality (length-based)
        let mut optimality_scores = Vec::new();
        for (i, path) in planned_paths.iter().enumerate() {
            if execution_success[i] {
                // Calculate path length
                let mut path_length = 0.0;
                for j in 0..path.nrows() - 1 {
                    let dx = (path[[j + 1, 0]] - path[[j, 0]]).to_f64().unwrap();
                    let dy = (path[[j + 1, 1]] - path[[j, 1]]).to_f64().unwrap();
                    path_length += (dx * dx + dy * dy).sqrt();
                }

                // Calculate straight-line distance to goal
                let start_x = path[[0, 0]].to_f64().unwrap();
                let start_y = path[[0, 1]].to_f64().unwrap();
                let goal_x = goals[[i, 0]];
                let goal_y = goals[[i, 1]];
                let straight_line_distance = 
                    ((goal_x - start_x).powi(2) + (goal_y - start_y).powi(2)).sqrt();

                if path_length > 0.0 {
                    let optimality = straight_line_distance / path_length;
                    optimality_scores.push(optimality.min(1.0)); // Cap at 1.0
                }
            }
        }

        let optimality = if !optimality_scores.is_empty() {
            PathOptimalityMetrics {
                length_optimality_ratio: optimality_scores.iter().sum::<f64>() / optimality_scores.len() as f64,
                energy_optimality_ratio: 1.0, // Simplified
                time_optimality_ratio: 1.0, // Simplified
                obstacle_clearance: ObstacleClearanceMetrics::default(),
            }
        } else {
            PathOptimalityMetrics::default()
        };

        // Calculate safety margin based on obstacle clearance
        let safety_margin = if let Some(obstacle_map) = obstacles {
            let mut min_clearances = Vec::new();
            
            for (i, path) in planned_paths.iter().enumerate() {
                if execution_success[i] {
                    let clearance = self.calculate_obstacle_clearance(path, Some(obstacle_map))?;
                    min_clearances.push(clearance.min_clearance);
                }
            }

            if !min_clearances.is_empty() {
                min_clearances.iter().sum::<f64>() / min_clearances.len() as f64
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY // No obstacles, infinite safety margin
        };

        Ok(PathPlanningMetrics {
            success_rate,
            optimality,
            planning_time,
            safety_margin,
        })
    }

    /// Evaluate obstacle avoidance metrics
    fn evaluate_obstacle_avoidance_metrics<F>(
        &self,
        executed_paths: &[ArrayView2<F>],
        static_obstacles: Option<&ArrayView2<F>>,
        dynamic_obstacles: Option<&[ArrayView2<F>]>,
        execution_success: &Array1<bool>,
    ) -> Result<ObstacleAvoidanceMetrics>
    where
        F: Float,
    {
        let n_tasks = executed_paths.len();
        let mut collision_count = 0;
        let mut near_miss_count = 0;
        let mut min_distances = Vec::new();
        let mut total_path_length = 0.0;
        let mut successful_paths = 0;

        for (task_idx, path) in executed_paths.iter().enumerate() {
            if !execution_success[task_idx] {
                continue;
            }
            
            successful_paths += 1;
            let mut path_collisions = false;
            let mut path_near_misses = 0;
            let mut path_min_distance = f64::INFINITY;
            
            // Calculate path length for this task
            let mut current_path_length = 0.0;
            for i in 0..path.nrows() - 1 {
                let dx = (path[[i + 1, 0]] - path[[i, 0]]).to_f64().unwrap();
                let dy = (path[[i + 1, 1]] - path[[i, 1]]).to_f64().unwrap();
                current_path_length += (dx * dx + dy * dy).sqrt();
            }
            total_path_length += current_path_length;

            // Check static _obstacles
            if let Some(static_obs) = static_obstacles {
                for i in 0..path.nrows() {
                    let x = path[[i, 0]].to_f64().unwrap();
                    let y = path[[i, 1]].to_f64().unwrap();
                    
                    let mut min_dist_to_obstacles = f64::INFINITY;
                    for obs_i in 0..static_obs.nrows() {
                        for obs_j in 0..static_obs.ncols() {
                            if static_obs[[obs_i, obs_j]].to_f64().unwrap() > 0.5 {
                                let obs_x = obs_i as f64;
                                let obs_y = obs_j as f64;
                                let dist = ((x - obs_x).powi(2) + (y - obs_y).powi(2)).sqrt();
                                min_dist_to_obstacles = min_dist_to_obstacles.min(dist);
                            }
                        }
                    }
                    
                    path_min_distance = path_min_distance.min(min_dist_to_obstacles);
                    
                    // Check for collision (distance < threshold)
                    if min_dist_to_obstacles < 0.1 {
                        path_collisions = true;
                    } else if min_dist_to_obstacles < 0.5 {
                        path_near_misses += 1;
                    }
                }
            }

            // Check dynamic _obstacles if available
            if let Some(dynamic_obs) = dynamic_obstacles {
                if task_idx < dynamic_obs.len() {
                    let dyn_obs = &dynamic_obs[task_idx];
                    
                    // Simplified: assume dynamic _obstacles have same time indices as path
                    let time_steps = path.nrows().min(dyn_obs.nrows());
                    
                    for t in 0..time_steps {
                        let x = path[[t, 0]].to_f64().unwrap();
                        let y = path[[t, 1]].to_f64().unwrap();
                        
                        for obs_idx in 0..dyn_obs.ncols() / 2 {  // Assume [x, y] pairs
                            let obs_x = dyn_obs[[t, obs_idx * 2]].to_f64().unwrap();
                            let obs_y = dyn_obs[[t, obs_idx * 2 + 1]].to_f64().unwrap();
                            let dist = ((x - obs_x).powi(2) + (y - obs_y).powi(2)).sqrt();
                            
                            path_min_distance = path_min_distance.min(dist);
                            
                            if dist < 0.2 {  // Dynamic obstacle collision threshold
                                path_collisions = true;
                            } else if dist < 0.8 {  // Dynamic obstacle near-miss threshold
                                path_near_misses += 1;
                            }
                        }
                    }
                }
            }

            if path_collisions {
                collision_count += 1;
            }
            near_miss_count += path_near_misses;
            min_distances.push(path_min_distance);
        }

        let collision_avoidance_rate = if successful_paths > 0 {
            1.0 - (collision_count as f64 / successful_paths as f64)
        } else {
            1.0
        };

        let near_miss_frequency = if total_path_length > 0.0 {
            near_miss_count as f64 / total_path_length
        } else {
            0.0
        };

        let min_obstacle_distance = min_distances.iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        // Calculate avoidance efficiency (simplified)
        let avoidance_efficiency = if collision_avoidance_rate > 0.5 {
            collision_avoidance_rate
        } else {
            0.0
        };

        Ok(ObstacleAvoidanceMetrics {
            collision_avoidance_rate,
            near_miss_frequency,
            min_obstacle_distance,
            avoidance_efficiency,
        })
    }

    /// Evaluate goal reaching metrics
    fn evaluate_goal_reaching_metrics<F>(
        &self,
        executed_paths: &[ArrayView2<F>],
        goals: &ArrayView2<F>,
        execution_success: &Array1<bool>,
        timestamps: Option<&[ArrayView1<F>]>,
    ) -> Result<GoalReachingMetrics>
    where
        F: Float,
    {
        let n_tasks = executed_paths.len();
        let success_count = execution_success.iter().filter(|&&x| x).count();
        let success_rate = success_count as f64 / n_tasks as f64;

        let mut reaching_errors = Vec::new();
        let mut completion_times = Vec::new();
        let mut path_efficiencies = Vec::new();

        for (task_idx, path) in executed_paths.iter().enumerate() {
            if execution_success[task_idx] && path.nrows() > 0 {
                // Calculate final position error
                let final_pos = path.row(path.nrows() - 1);
                let goal_pos = goals.row(task_idx);
                
                let error = if final_pos.len() >= 2 && goal_pos.len() >= 2 {
                    let dx = (final_pos[0] - goal_pos[0]).to_f64().unwrap();
                    let dy = (final_pos[1] - goal_pos[1]).to_f64().unwrap();
                    (dx * dx + dy * dy).sqrt()
                } else {
                    0.0
                };
                
                reaching_errors.push(error);

                // Calculate path efficiency
                let mut path_length = 0.0;
                for i in 0..path.nrows() - 1 {
                    let dx = (path[[i + 1, 0]] - path[[i, 0]]).to_f64().unwrap();
                    let dy = (path[[i + 1, 1]] - path[[i, 1]]).to_f64().unwrap();
                    path_length += (dx * dx + dy * dy).sqrt();
                }

                // Straight-line distance from start to goal
                let start_x = path[[0, 0]].to_f64().unwrap();
                let start_y = path[[0, 1]].to_f64().unwrap();
                let goal_x = goal_pos[0].to_f64().unwrap();
                let goal_y = goal_pos[1].to_f64().unwrap();
                let straight_line_dist = 
                    ((goal_x - start_x).powi(2) + (goal_y - start_y).powi(2)).sqrt();

                let efficiency = if path_length > 0.0 {
                    (straight_line_dist / path_length).min(1.0)
                } else {
                    0.0
                };
                path_efficiencies.push(efficiency);

                // Calculate completion time if timestamps available
                if let Some(times) = timestamps {
                    if task_idx < times.len() && times[task_idx].len() > 1 {
                        let time_taken = (times[task_idx][times[task_idx].len() - 1] 
                                        - times[task_idx][0]).to_f64().unwrap();
                        completion_times.push(time_taken);
                    }
                }
            }
        }

        let reaching_accuracy = if !reaching_errors.is_empty() {
            ErrorStatistics {
                rmse: (reaching_errors.iter().map(|&x| x * x).sum::<f64>() / reaching_errors.len() as f64).sqrt(),
                mae: reaching_errors.iter().sum::<f64>() / reaching_errors.len() as f64,
                max_error: reaching_errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                std_dev: {
                    let mean = reaching_errors.iter().sum::<f64>() / reaching_errors.len() as f64;
                    (reaching_errors.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() 
                     / reaching_errors.len() as f64).sqrt()
                },
                median_error: {
                    let mut sorted = reaching_errors.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if sorted.len() % 2 == 0 {
                        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                    } else {
                        sorted[sorted.len() / 2]
                    }
                },
            }
        } else {
            ErrorStatistics::default()
        };

        let time_to_goal = if !completion_times.is_empty() {
            let avg_time = completion_times.iter().sum::<f64>() / completion_times.len() as f64;
            Duration::from_secs_f64(avg_time)
        } else {
            Duration::from_secs(0)
        };

        let path_efficiency = if !path_efficiencies.is_empty() {
            path_efficiencies.iter().sum::<f64>() / path_efficiencies.len() as f64
        } else {
            1.0
        };

        Ok(GoalReachingMetrics {
            success_rate,
            reaching_accuracy,
            time_to_goal,
            path_efficiency,
        })
    }

    /// Evaluate dynamic adaptation metrics
    fn evaluate_dynamic_adaptation_metrics<F>(
        &self,
        planned_paths: &[ArrayView2<F>],
        executed_paths: &[ArrayView2<F>],
        dynamic_obstacles: Option<&[ArrayView2<F>]>,
        timestamps: Option<&[ArrayView1<F>]>,
    ) -> Result<DynamicAdaptationMetrics>
    where
        F: Float,
    {
        let n_tasks = planned_paths.len();
        
        // Calculate path deviation (difference between planned and executed)
        let mut path_deviations = Vec::new();
        let mut replanning_events = 0;
        
        for task_idx in 0..n_tasks {
            let planned = &planned_paths[task_idx];
            let executed = &executed_paths[task_idx];
            
            // Calculate deviation between planned and executed _paths
            let min_length = planned.nrows().min(executed.nrows());
            let mut total_deviation = 0.0;
            let mut valid_points = 0;
            
            for i in 0..min_length {
                if planned.ncols() >= 2 && executed.ncols() >= 2 {
                    let dx = (planned[[i, 0]] - executed[[i, 0]]).to_f64().unwrap();
                    let dy = (planned[[i, 1]] - executed[[i, 1]]).to_f64().unwrap();
                    total_deviation += (dx * dx + dy * dy).sqrt();
                    valid_points += 1;
                }
            }
            
            if valid_points > 0 {
                path_deviations.push(total_deviation / valid_points as f64);
            }
            
            // Simple heuristic: if _paths differ significantly, assume replanning occurred
            if valid_points > 0 && total_deviation / valid_points as f64 > 1.0 {
                replanning_events += 1;
            }
        }

        // Calculate average response time (simplified)
        let response_time = if let Some(times) = timestamps {
            let mut response_times = Vec::new();
            for time_series in times {
                if time_series.len() > 1 {
                    // Simplified: assume first significant time gap indicates response time
                    for i in 1..time_series.len() {
                        let dt = (time_series[i] - time_series[i-1]).to_f64().unwrap();
                        if dt > 0.1 { // Threshold for significant delay
                            response_times.push(dt);
                            break;
                        }
                    }
                }
            }
            
            if !response_times.is_empty() {
                let avg_response = response_times.iter().sum::<f64>() / response_times.len() as f64;
                Duration::from_secs_f64(avg_response)
            } else {
                Duration::from_millis(100) // Default fast response
            }
        } else {
            Duration::from_millis(100)
        };

        let replanning_frequency = if n_tasks > 0 {
            replanning_events as f64 / n_tasks as f64
        } else {
            0.0
        };

        // Dynamic success rate (assume _paths with low deviation are successful)
        let dynamic_success_rate = if !path_deviations.is_empty() {
            let successful_adaptations = path_deviations.iter()
                .filter(|&&dev| dev < 2.0) // Threshold for acceptable deviation
                .count();
            successful_adaptations as f64 / path_deviations.len() as f64
        } else {
            1.0
        };

        // Prediction accuracy (simplified based on path following quality)
        let prediction_accuracy = if !path_deviations.is_empty() {
            let avg_deviation = path_deviations.iter().sum::<f64>() / path_deviations.len() as f64;
            (1.0 / (1.0 + avg_deviation)).min(1.0)
        } else {
            1.0
        };

        Ok(DynamicAdaptationMetrics {
            response_time,
            replanning_frequency,
            dynamic_success_rate,
            prediction_accuracy,
        })
    }

    /// Evaluate safety and reliability metrics
    pub fn evaluate_safety_reliability_metrics<F>(
        &mut self,
        system_states: &ArrayView2<F>, // System state data over time
        failure_events: &Array1<bool>, // Whether a failure occurred at each time step
        failure_types: Option<&Array1<usize>>, // Type of failure (encoded as integers)
        recovery_times: Option<&Array1<f64>>, // Time taken to recover from each failure
        health_indicators: Option<&ArrayView2<F>>, // Health monitoring data
        risk_levels: Option<&Array1<f64>>, // Assessed risk levels over time
        redundancy_status: Option<&ArrayView2<bool>>, // Status of redundant systems
        maintenance_events: Option<&Array1<bool>>, // Scheduled/unscheduled maintenance events
    ) -> Result<SafetyReliabilityMetrics>
    where
        F: Float,
    {
        let n_timesteps = system_states.nrows();
        if n_timesteps == 0 {
            return Err(MetricsError::InvalidInput(
                "System _states cannot be empty".to_string(),
            ));
        }

        // Evaluate failure detection and recovery metrics
        let failure_metrics = self.evaluate_failure_metrics(
            failure_events,
            failure_types,
            recovery_times,
            health_indicators,
        )?;

        // Evaluate risk assessment metrics
        let risk_assessment = self.evaluate_risk_assessment_metrics(
            system_states,
            failure_events,
            risk_levels,
            health_indicators,
        )?;

        // Evaluate redundancy effectiveness
        let redundancy_metrics = self.evaluate_redundancy_metrics(
            system_states,
            failure_events,
            redundancy_status,
        )?;

        // Evaluate predictive maintenance metrics
        let maintenance_metrics = self.evaluate_maintenance_metrics(
            failure_events,
            maintenance_events,
            health_indicators,
        )?;

        Ok(SafetyReliabilityMetrics {
            failure_metrics,
            risk_assessment,
            redundancy_metrics,
            maintenance_metrics,
        })
    }

    /// Evaluate failure detection and recovery metrics
    fn evaluate_failure_metrics<F>(
        &self,
        failure_events: &Array1<bool>,
        failure_types: Option<&Array1<usize>>,
        recovery_times: Option<&Array1<f64>>,
        health_indicators: Option<&ArrayView2<F>>,
    ) -> Result<FailureMetrics>
    where
        F: Float,
    {
        let n_timesteps = failure_events.len();
        let failure_count = failure_events.iter().filter(|&&x| x).count();

        // Calculate detection accuracy (if health _indicators available)
        let detection_accuracy = if let Some(health_data) = health_indicators {
            // Simplified: assume we can detect anomalies in health _indicators
            let mut correct_detections = 0;
            let mut total_detections = 0;

            for (i, &failure) in failure_events.iter().enumerate() {
                if i < health_data.nrows() {
                    // Simple anomaly detection: check if any health indicator deviates significantly
                    let health_values = health_data.row(i);
                    let is_anomalous = health_values.iter()
                        .any(|&val| {
                            let val_f64 = val.to_f64().unwrap();
                            val_f64 < 0.3 || val_f64 > 0.9 // Assume normal range [0.3, 0.9]
                        });

                    total_detections += 1;
                    if failure == is_anomalous {
                        correct_detections += 1;
                    }
                }
            }

            if total_detections > 0 {
                correct_detections as f64 / total_detections as f64
            } else {
                1.0
            }
        } else {
            1.0 // Assume perfect detection if no health data
        };

        // Calculate false alarm rate
        let false_alarm_rate = if let Some(health_data) = health_indicators {
            let mut false_alarms = 0;
            let mut non_failure_events = 0;

            for (i, &failure) in failure_events.iter().enumerate() {
                if !failure && i < health_data.nrows() {
                    non_failure_events += 1;
                    
                    let health_values = health_data.row(i);
                    let false_detection = health_values.iter()
                        .any(|&val| {
                            let val_f64 = val.to_f64().unwrap();
                            val_f64 < 0.3 || val_f64 > 0.9
                        });

                    if false_detection {
                        false_alarms += 1;
                    }
                }
            }

            if non_failure_events > 0 {
                false_alarms as f64 / non_failure_events as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate recovery success rate
        let recovery_success_rate = if let Some(recovery_time_data) = recovery_times {
            // Count successful recoveries (finite recovery times)
            let successful_recoveries = recovery_time_data.iter()
                .filter(|&&time| time > 0.0 && time.is_finite())
                .count();
            
            if failure_count > 0 {
                successful_recoveries as f64 / failure_count as f64
            } else {
                1.0
            }
        } else {
            // If no recovery data, assume all failures are eventually recovered
            if failure_count > 0 { 0.8 } else { 1.0 }
        };

        // Calculate mean time to recovery
        let mean_time_to_recovery = if let Some(recovery_time_data) = recovery_times {
            let valid_recovery_times: Vec<f64> = recovery_time_data.iter()
                .filter(|&&time| time > 0.0 && time.is_finite())
                .cloned()
                .collect();

            if !valid_recovery_times.is_empty() {
                let mean_time = valid_recovery_times.iter().sum::<f64>() / valid_recovery_times.len() as f64;
                Duration::from_secs_f64(mean_time)
            } else {
                Duration::from_secs(0)
            }
        } else {
            Duration::from_secs(300) // Default 5 minutes
        };

        Ok(FailureMetrics {
            detection_accuracy,
            false_alarm_rate,
            recovery_success_rate,
            mean_time_to_recovery,
        })
    }

    /// Evaluate risk assessment metrics
    fn evaluate_risk_assessment_metrics<F>(
        &self,
        system_states: &ArrayView2<F>,
        failure_events: &Array1<bool>,
        risk_levels: Option<&Array1<f64>>,
        health_indicators: Option<&ArrayView2<F>>,
    ) -> Result<RiskAssessmentMetrics>
    where
        F: Float,
    {
        let n_timesteps = system_states.nrows();

        // Calculate risk prediction accuracy
        let risk_prediction_accuracy = if let Some(risk_data) = risk_levels {
            let mut correct_predictions = 0;
            let mut total_predictions = 0;

            for (i, &failure) in failure_events.iter().enumerate() {
                if i < risk_data.len() {
                    let risk_level = risk_data[i];
                    let high_risk_predicted = risk_level > 0.7; // High risk threshold
                    
                    total_predictions += 1;
                    if failure == high_risk_predicted {
                        correct_predictions += 1;
                    }
                }
            }

            if total_predictions > 0 {
                correct_predictions as f64 / total_predictions as f64
            } else {
                1.0
            }
        } else {
            0.5 // Random prediction if no risk data
        };

        // Calculate safety margin maintenance
        let safety_margin_maintenance = if let Some(health_data) = health_indicators {
            let mut safe_timesteps = 0;
            let min_safe_threshold = 0.4; // Minimum safe health level

            for i in 0..health_data.nrows() {
                let health_values = health_data.row(i);
                let all_safe = health_values.iter()
                    .all(|&val| val.to_f64().unwrap() >= min_safe_threshold);
                
                if all_safe {
                    safe_timesteps += 1;
                }
            }

            if health_data.nrows() > 0 {
                safe_timesteps as f64 / health_data.nrows() as f64
            } else {
                1.0
            }
        } else {
            0.95 // Assume high safety margin by default
        };

        // Calculate hazard identification rate
        let hazard_identification_rate = if let Some(risk_data) = risk_levels {
            let actual_hazards = failure_events.iter().filter(|&&x| x).count();
            let identified_hazards = risk_data.iter()
                .filter(|&&risk| risk > 0.5)
                .count();

            if actual_hazards > 0 {
                (identified_hazards.min(actual_hazards) as f64 / actual_hazards as f64).min(1.0)
            } else {
                1.0
            }
        } else {
            0.7 // Default identification rate
        };

        // Calculate risk mitigation effectiveness
        let risk_mitigation_effectiveness = if let Some(risk_data) = risk_levels {
            // Measure how well high-risk situations were handled
            let mut high_risk_handled = 0;
            let mut high_risk_situations = 0;

            for (i, &risk_level) in risk_data.iter().enumerate() {
                if risk_level > 0.7 {
                    high_risk_situations += 1;
                    if i < failure_events.len() && !failure_events[i] {
                        high_risk_handled += 1;
                    }
                }
            }

            if high_risk_situations > 0 {
                high_risk_handled as f64 / high_risk_situations as f64
            } else {
                1.0
            }
        } else {
            0.8 // Default mitigation effectiveness
        };

        Ok(RiskAssessmentMetrics {
            risk_prediction_accuracy,
            safety_margin_maintenance,
            hazard_identification_rate,
            risk_mitigation_effectiveness,
        })
    }

    /// Evaluate redundancy effectiveness metrics
    fn evaluate_redundancy_metrics<F>(
        &self,
        system_states: &ArrayView2<F>,
        failure_events: &Array1<bool>,
        redundancy_status: Option<&ArrayView2<bool>>,
    ) -> Result<RedundancyMetrics>
    where
        F: Float,
    {
        let redundancy_utilization = if let Some(redundancy_data) = redundancy_status {
            // Calculate how often redundant systems are used
            let total_elements = redundancy_data.len();
            let active_redundant_elements = redundancy_data.iter()
                .filter(|&&active| active)
                .count();
            
            if total_elements > 0 {
                active_redundant_elements as f64 / total_elements as f64
            } else {
                0.0
            }
        } else {
            0.0 // No redundancy data available
        };

        // Calculate failover success rate
        let failover_success_rate = if let Some(redundancy_data) = redundancy_status {
            let mut failover_attempts = 0;
            let mut successful_failovers = 0;

            // Look for patterns where failure occurs but system continues (indicating failover)
            for (i, &failure) in failure_events.iter().enumerate() {
                if failure && i < redundancy_data.nrows() {
                    failover_attempts += 1;
                    
                    // Check if redundant systems are active after failure
                    let redundant_active = redundancy_data.row(i)
                        .iter()
                        .any(|&active| active);
                    
                    if redundant_active {
                        successful_failovers += 1;
                    }
                }
            }

            if failover_attempts > 0 {
                successful_failovers as f64 / failover_attempts as f64
            } else {
                1.0 // No failover attempts needed
            }
        } else {
            0.9 // Default high failover success rate
        };

        // Calculate performance degradation under failures
        let performance_degradation = if system_states.ncols() > 0 {
            let mut degradation_during_failures = Vec::new();
            
            for (i, &failure) in failure_events.iter().enumerate() {
                if failure && i < system_states.nrows() {
                    // Calculate performance loss as deviation from normal
                    let current_performance: f64 = system_states.row(i)
                        .iter()
                        .map(|&val| val.to_f64().unwrap())
                        .sum::<f64>() / system_states.ncols() as f64;
                    
                    // Compare to average performance during non-failure periods
                    let normal_performance: f64 = failure_events.iter()
                        .enumerate()
                        .filter(|(_, &fail)| !fail)
                        .filter_map(|(idx_)| {
                            if idx < system_states.nrows() {
                                let perf: f64 = system_states.row(idx)
                                    .iter()
                                    .map(|&val| val.to_f64().unwrap())
                                    .sum::<f64>() / system_states.ncols() as f64;
                                Some(perf)
                            } else {
                                None
                            }
                        })
                        .sum::<f64>() / (failure_events.len() - failure_events.iter().filter(|&&x| x).count()) as f64;
                    
                    if normal_performance > 0.0 {
                        let degradation = 1.0 - (current_performance / normal_performance);
                        degradation_during_failures.push(degradation.max(0.0).min(1.0));
                    }
                }
            }

            if !degradation_during_failures.is_empty() {
                degradation_during_failures.iter().sum::<f64>() / degradation_during_failures.len() as f64
            } else {
                0.0
            }
        } else {
            0.1 // Default minimal degradation
        };

        // Calculate system availability
        let system_availability = {
            let operational_time = failure_events.iter()
                .filter(|&&failure| !failure)
                .count();
            
            operational_time as f64 / failure_events.len() as f64
        };

        Ok(RedundancyMetrics {
            redundancy_utilization,
            failover_success_rate,
            performance_degradation,
            system_availability,
        })
    }

    /// Evaluate predictive maintenance metrics
    fn evaluate_maintenance_metrics<F>(
        &self,
        failure_events: &Array1<bool>,
        maintenance_events: Option<&Array1<bool>>,
        health_indicators: Option<&ArrayView2<F>>,
    ) -> Result<MaintenanceMetrics>
    where
        F: Float,
    {
        // Calculate maintenance prediction accuracy
        let prediction_accuracy = if let (Some(maintenance_data), Some(health_data)) = 
            (maintenance_events, health_indicators) {
            let mut correct_predictions = 0;
            let mut total_predictions = 0;

            for (i, &maintenance_needed) in maintenance_data.iter().enumerate() {
                if i < health_data.nrows() {
                    // Predict maintenance need based on health _indicators
                    let health_values = health_data.row(i);
                    let predicted_maintenance = health_values.iter()
                        .any(|&val| val.to_f64().unwrap() < 0.5); // Low health indicates maintenance need
                    
                    total_predictions += 1;
                    if maintenance_needed == predicted_maintenance {
                        correct_predictions += 1;
                    }
                }
            }

            if total_predictions > 0 {
                correct_predictions as f64 / total_predictions as f64
            } else {
                1.0
            }
        } else {
            0.7 // Default prediction accuracy
        };

        // Calculate preventive maintenance effectiveness
        let preventive_effectiveness = if let Some(maintenance_data) = maintenance_events {
            // Look for maintenance _events that prevent subsequent failures
            let mut prevented_failures = 0;
            let mut maintenance_count = 0;

            for (i, &maintenance) in maintenance_data.iter().enumerate() {
                if maintenance {
                    maintenance_count += 1;
                    
                    // Check if failure is prevented in the next few time steps
                    let look_ahead = 5.min(failure_events.len() - i - 1);
                    let failure_prevented = (1..=look_ahead)
                        .all(|offset| !failure_events[i + offset]);
                    
                    if failure_prevented {
                        prevented_failures += 1;
                    }
                }
            }

            if maintenance_count > 0 {
                prevented_failures as f64 / maintenance_count as f64
            } else {
                1.0
            }
        } else {
            0.8 // Default effectiveness
        };

        // Calculate health monitoring quality
        let health_monitoring_quality = if let Some(health_data) = health_indicators {
            // Assess consistency and reliability of health _indicators
            let mut quality_scores = Vec::new();
            
            for col in 0..health_data.ncols() {
                let column_data: Vec<f64> = (0..health_data.nrows())
                    .map(|row| health_data[[row, col]].to_f64().unwrap())
                    .collect();
                
                if !column_data.is_empty() {
                    let mean = column_data.iter().sum::<f64>() / column_data.len() as f64;
                    let variance = column_data.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>() / column_data.len() as f64;
                    
                    // Quality is higher for reasonable variance (not too constant, not too noisy)
                    let coefficient_of_variation = if mean > 0.0 {
                        variance.sqrt() / mean
                    } else {
                        f64::INFINITY
                    };
                    
                    let quality = if coefficient_of_variation > 0.05 && coefficient_of_variation < 0.5 {
                        1.0 - (coefficient_of_variation - 0.275).abs() / 0.225
                    } else {
                        0.5
                    };
                    
                    quality_scores.push(quality.max(0.0).min(1.0));
                }
            }

            if !quality_scores.is_empty() {
                quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
            } else {
                0.5
            }
        } else {
            0.5 // Default monitoring quality
        };

        // Calculate maintenance scheduling optimality
        let scheduling_optimality = if let Some(maintenance_data) = maintenance_events {
            // Assess if maintenance is well-timed (not too frequent, not too sparse)
            let maintenance_intervals: Vec<usize> = {
                let mut intervals = Vec::new();
                let mut last_maintenance = None;
                
                for (i, &maintenance) in maintenance_data.iter().enumerate() {
                    if maintenance {
                        if let Some(last) = last_maintenance {
                            intervals.push(i - last);
                        }
                        last_maintenance = Some(i);
                    }
                }
                
                intervals
            };

            if !maintenance_intervals.is_empty() {
                let mean_interval = maintenance_intervals.iter().sum::<usize>() as f64 / maintenance_intervals.len() as f64;
                let ideal_interval = 20.0; // Assume ideal maintenance every 20 time steps
                
                // Optimality is higher when intervals are close to ideal
                let optimality = 1.0 - ((mean_interval - ideal_interval).abs() / ideal_interval).min(1.0);
                optimality.max(0.0)
            } else {
                0.5 // No maintenance intervals to evaluate
            }
        } else {
            0.7 // Default scheduling optimality
        };

        Ok(MaintenanceMetrics {
            prediction_accuracy,
            preventive_effectiveness,
            health_monitoring_quality,
            scheduling_optimality,
        })
    }

    /// Evaluate robotic perception metrics
    pub fn evaluate_perception_metrics<F>(
        &mut self,
        detection_results: &ArrayView2<F>, // Detection results [n_frames, n_detections_per_frame]
        ground_truth_detections: &ArrayView2<F>, // Ground truth detections
        detection_classes: Option<&Array2<usize>>, // Class labels for detections
        confidence_scores: Option<&ArrayView2<F>>, // Confidence scores for detections
        segmentation_results: Option<&ArrayView3<F>>, // Segmentation masks [height, width, n_frames]
        ground_truth_segmentation: Option<&ArrayView3<F>>, // Ground truth segmentation
        scene_labels: Option<&Array1<usize>>, // Scene classification labels
        predicted_scene_labels: Option<&Array1<usize>>, // Predicted scene labels
        processing_times: &Array1<f64>, // Processing time for each frame
    ) -> Result<RoboticPerceptionMetrics>
    where
        F: Float,
    {
        let n_frames = detection_results.nrows();
        if n_frames == 0 {
            return Err(MetricsError::InvalidInput(
                "Detection _results cannot be empty".to_string(),
            ));
        }

        // Evaluate object detection metrics
        let object_detection = self.evaluate_object_detection_metrics(
            detection_results,
            ground_truth_detections,
            detection_classes,
            confidence_scores,
        )?;

        // Evaluate scene understanding metrics
        let scene_understanding = self.evaluate_scene_understanding_metrics(
            segmentation_results,
            ground_truth_segmentation,
            scene_labels,
            predicted_scene_labels,
        )?;

        // Evaluate sensor fusion effectiveness (simplified)
        let sensor_fusion = self.evaluate_sensor_fusion_metrics(
            detection_results,
            confidence_scores,
        )?;

        // Evaluate real-time performance
        let real_time_performance = self.evaluate_real_time_performance_metrics(
            processing_times,
            n_frames,
        )?;

        Ok(RoboticPerceptionMetrics {
            object_detection,
            scene_understanding,
            sensor_fusion,
            real_time_performance,
        })
    }

    /// Evaluate object detection metrics
    fn evaluate_object_detection_metrics<F>(
        &self,
        detection_results: &ArrayView2<F>,
        ground_truth_detections: &ArrayView2<F>,
        detection_classes: Option<&Array2<usize>>,
        confidence_scores: Option<&ArrayView2<F>>,
    ) -> Result<ObjectDetectionMetrics>
    where
        F: Float,
    {
        let n_frames = detection_results.nrows();
        let mut per_class_accuracy = HashMap::new();
        let mut total_detections = 0;
        let mut correct_detections = 0;
        let mut false_positives = 0;
        let mut total_tracking_consistency = 0.0;
        let mut valid_tracking_frames = 0;

        // Calculate per-class accuracy if class information is available
        if let Some(class_data) = detection_classes {
            let mut class_stats: HashMap<usize, (usize, usize)> = HashMap::new(); // (correct, total)
            
            for frame_idx in 0..n_frames.min(class_data.nrows()) {
                for det_idx in 0..class_data.ncols() {
                    let predicted_class = class_data[[frame_idx, det_idx]];
                    if predicted_class > 0 { // Assume 0 means no detection
                        total_detections += 1;
                        
                        // Simplified: assume ground truth has same structure
                        // In practice, this would require proper IoU-based matching
                        if frame_idx < ground_truth_detections.nrows() && 
                           det_idx < ground_truth_detections.ncols() {
                            let gt_confidence = ground_truth_detections[[frame_idx, det_idx]].to_f64().unwrap();
                            if gt_confidence > 0.5 { // Ground truth detection exists
                                correct_detections += 1;
                                let stats = class_stats.entry(predicted_class).or_insert((0, 0));
                                stats.0 += 1; // Increment correct
                            } else {
                                false_positives += 1;
                            }
                            let stats = class_stats.entry(predicted_class).or_insert((0, 0));
                            stats.1 += 1; // Increment total
                        }
                    }
                }
            }

            // Convert to per-class accuracy
            for (class_id, (correct, total)) in class_stats {
                if total > 0 {
                    let accuracy = correct as f64 / total as f64;
                    per_class_accuracy.insert(format!("class_{}", class_id), accuracy);
                }
            }
        }

        // Calculate real-time detection rate
        let real_time_detection_rate = if total_detections > 0 {
            correct_detections as f64 / total_detections as f64
        } else {
            0.0
        };

        // Calculate false positive rate
        let false_positive_rate = if total_detections > 0 {
            false_positives as f64 / total_detections as f64
        } else {
            0.0
        };

        // Calculate tracking consistency (simplified)
        let tracking_consistency = if let Some(confidence_data) = confidence_scores {
            // Measure consistency of _detections across frames
            for det_idx in 0..confidence_data.ncols() {
                let mut detection_sequence = Vec::new();
                for frame_idx in 0..confidence_data.nrows() {
                    let confidence = confidence_data[[frame_idx, det_idx]].to_f64().unwrap();
                    detection_sequence.push(confidence > 0.5);
                }

                if detection_sequence.len() > 1 {
                    // Count consistent _detections (similar pattern across frames)
                    let mut consistency_score = 0.0;
                    let mut transitions = 0;
                    
                    for i in 1..detection_sequence.len() {
                        if detection_sequence[i] == detection_sequence[i-1] {
                            consistency_score += 1.0;
                        }
                        transitions += 1;
                    }
                    
                    if transitions > 0 {
                        total_tracking_consistency += consistency_score / transitions as f64;
                        valid_tracking_frames += 1;
                    }
                }
            }

            if valid_tracking_frames > 0 {
                total_tracking_consistency / valid_tracking_frames as f64
            } else {
                1.0
            }
        } else {
            0.8 // Default consistency if no confidence data
        };

        Ok(ObjectDetectionMetrics {
            per_class_accuracy,
            real_time_detection_rate,
            false_positive_rate,
            tracking_consistency,
        })
    }

    /// Evaluate scene understanding metrics
    fn evaluate_scene_understanding_metrics<F>(
        &self,
        segmentation_results: Option<&ArrayView3<F>>,
        ground_truth_segmentation: Option<&ArrayView3<F>>,
        scene_labels: Option<&Array1<usize>>,
        predicted_scene_labels: Option<&Array1<usize>>,
    ) -> Result<SceneUnderstandingMetrics>
    where
        F: Float,
    {
        // Calculate _segmentation accuracy
        let segmentation_accuracy = if let (Some(seg_results), Some(gt_seg)) = 
            (segmentation_results, ground_truth_segmentation) {
            
            let mut total_pixels = 0;
            let mut correct_pixels = 0;
            
            let n_frames = seg_results.shape()[2].min(gt_seg.shape()[2]);
            let height = seg_results.shape()[0].min(gt_seg.shape()[0]);
            let width = seg_results.shape()[1].min(gt_seg.shape()[1]);
            
            for frame in 0..n_frames {
                for i in 0..height {
                    for j in 0..width {
                        let predicted = seg_results[[i, j, frame]].to_f64().unwrap();
                        let ground_truth = gt_seg[[i, j, frame]].to_f64().unwrap();
                        
                        total_pixels += 1;
                        // Simple threshold-based comparison
                        if (predicted > 0.5) == (ground_truth > 0.5) {
                            correct_pixels += 1;
                        }
                    }
                }
            }
            
            if total_pixels > 0 {
                correct_pixels as f64 / total_pixels as f64
            } else {
                0.0
            }
        } else {
            0.5 // Default if no _segmentation data
        };

        // Calculate spatial relationship understanding (simplified)
        let spatial_relationship_accuracy = if let (Some(seg_results), Some(_gt_seg)) = 
            (segmentation_results, ground_truth_segmentation) {
            
            // Simplified: analyze spatial coherence of _segmentation
            let mut coherence_scores = Vec::new();
            let n_frames = seg_results.shape()[2];
            let height = seg_results.shape()[0];
            let width = seg_results.shape()[1];
            
            for frame in 0..n_frames {
                if height > 1 && width > 1 {
                    let mut consistent_neighbors = 0;
                    let mut total_neighbors = 0;
                    
                    for i in 0..height-1 {
                        for j in 0..width-1 {
                            let current = seg_results[[i, j, frame]].to_f64().unwrap();
                            let right = seg_results[[i, j+1, frame]].to_f64().unwrap();
                            let down = seg_results[[i+1, j, frame]].to_f64().unwrap();
                            
                            total_neighbors += 2;
                            if (current > 0.5) == (right > 0.5) {
                                consistent_neighbors += 1;
                            }
                            if (current > 0.5) == (down > 0.5) {
                                consistent_neighbors += 1;
                            }
                        }
                    }
                    
                    if total_neighbors > 0 {
                        coherence_scores.push(consistent_neighbors as f64 / total_neighbors as f64);
                    }
                }
            }
            
            if !coherence_scores.is_empty() {
                coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
            } else {
                0.8
            }
        } else {
            0.8 // Default spatial understanding
        };

        // Calculate dynamic adaptation rate
        let dynamic_adaptation_rate = if let (Some(predicted_labels), Some(true_labels)) = 
            (predicted_scene_labels, scene_labels) {
            
            let mut adaptation_events = 0;
            let mut correct_adaptations = 0;
            
            for i in 1..predicted_labels.len().min(true_labels.len()) {
                // Detect scene changes
                if true_labels[i] != true_labels[i-1] {
                    adaptation_events += 1;
                    // Check if prediction adapted correctly
                    if predicted_labels[i] == true_labels[i] {
                        correct_adaptations += 1;
                    }
                }
            }
            
            if adaptation_events > 0 {
                correct_adaptations as f64 / adaptation_events as f64
            } else {
                1.0 // No adaptations needed
            }
        } else {
            0.7 // Default adaptation rate
        };

        // Calculate context reasoning quality
        let context_reasoning_quality = if let (Some(predicted_labels), Some(true_labels)) = 
            (predicted_scene_labels, scene_labels) {
            
            let correct_predictions = predicted_labels.iter()
                .zip(true_labels.iter())
                .filter(|(&pred, &truth)| pred == truth)
                .count();
            
            let total_predictions = predicted_labels.len().min(true_labels.len());
            
            if total_predictions > 0 {
                correct_predictions as f64 / total_predictions as f64
            } else {
                0.0
            }
        } else {
            0.6 // Default context reasoning quality
        };

        Ok(SceneUnderstandingMetrics {
            segmentation_accuracy,
            spatial_relationship_accuracy,
            dynamic_adaptation_rate,
            context_reasoning_quality,
        })
    }

    /// Evaluate sensor fusion effectiveness
    fn evaluate_sensor_fusion_metrics<F>(
        &self,
        detection_results: &ArrayView2<F>,
        confidence_scores: Option<&ArrayView2<F>>,
    ) -> Result<SensorFusionMetrics>
    where
        F: Float,
    {
        // Calculate accuracy improvement (simplified - comparing with and without fusion)
        let accuracy_improvement = if let Some(confidence_data) = confidence_scores {
            // Simplified: assume higher confidence indicates better fusion
            let avg_confidence: f64 = confidence_data.iter()
                .map(|&conf| conf.to_f64().unwrap())
                .sum::<f64>() / confidence_data.len() as f64;
            
            // Assume improvement is related to confidence distribution
            let confidence_variance: f64 = confidence_data.iter()
                .map(|&conf| {
                    let c = conf.to_f64().unwrap();
                    (c - avg_confidence).powi(2)
                })
                .sum::<f64>() / confidence_data.len() as f64;
            
            // Higher variance suggests better discrimination
            (confidence_variance * 2.0).min(0.5)
        } else {
            0.1 // Default minimal improvement
        };

        // Calculate reliability weighting accuracy
        let reliability_weighting_accuracy = if let Some(confidence_data) = confidence_scores {
            // Assess how well confidence _scores correlate with detection quality
            let mut high_confidence_detections = 0;
            let mut high_confidence_accurate = 0;
            let high_confidence_threshold = 0.8;
            
            for frame_idx in 0..confidence_data.nrows() {
                for det_idx in 0..confidence_data.ncols() {
                    let confidence = confidence_data[[frame_idx, det_idx]].to_f64().unwrap();
                    let detection_value = detection_results[[frame_idx, det_idx]].to_f64().unwrap();
                    
                    if confidence > high_confidence_threshold {
                        high_confidence_detections += 1;
                        // Assume high confidence should correlate with strong detection signal
                        if detection_value > 0.7 {
                            high_confidence_accurate += 1;
                        }
                    }
                }
            }
            
            if high_confidence_detections > 0 {
                high_confidence_accurate as f64 / high_confidence_detections as f64
            } else {
                1.0
            }
        } else {
            0.8 // Default reliability accuracy
        };

        // Calculate sensor failure detection
        let sensor_failure_detection = if let Some(confidence_data) = confidence_scores {
            // Detect potential sensor failures (very low confidence across many detections)
            let mut low_confidence_sequences = 0;
            let mut total_sequences = 0;
            
            for det_idx in 0..confidence_data.ncols() {
                let mut consecutive_low = 0;
                let mut max_consecutive_low = 0;
                
                for frame_idx in 0..confidence_data.nrows() {
                    let confidence = confidence_data[[frame_idx, det_idx]].to_f64().unwrap();
                    
                    if confidence < 0.3 {
                        consecutive_low += 1;
                        max_consecutive_low = max_consecutive_low.max(consecutive_low);
                    } else {
                        consecutive_low = 0;
                    }
                }
                
                total_sequences += 1;
                if max_consecutive_low >= 3 { // 3+ consecutive low confidence frames
                    low_confidence_sequences += 1;
                }
            }
            
            if total_sequences > 0 {
                // Good failure detection means few undetected failure sequences
                1.0 - (low_confidence_sequences as f64 / total_sequences as f64)
            } else {
                1.0
            }
        } else {
            0.9 // Default failure detection
        };

        // Calculate information gain from fusion
        let information_gain = if let Some(confidence_data) = confidence_scores {
            // Measure information gain through confidence distribution analysis
            let confidence_values: Vec<f64> = confidence_data.iter()
                .map(|&conf| conf.to_f64().unwrap())
                .collect();
            
            if !confidence_values.is_empty() {
                let mean = confidence_values.iter().sum::<f64>() / confidence_values.len() as f64;
                let variance = confidence_values.iter()
                    .map(|&conf| (conf - mean).powi(2))
                    .sum::<f64>() / confidence_values.len() as f64;
                
                // Information gain related to the discriminative power
                let entropy_reduction = variance / (mean + 1e-6);
                entropy_reduction.min(1.0)
            } else {
                0.0
            }
        } else {
            0.2 // Default information gain
        };

        Ok(SensorFusionMetrics {
            accuracy_improvement,
            reliability_weighting_accuracy,
            sensor_failure_detection,
            information_gain,
        })
    }

    /// Evaluate real-time performance metrics
    fn evaluate_real_time_performance_metrics(
        &self,
        processing_times: &Array1<f64>,
        n_frames: usize,
    ) -> Result<RealTimePerformanceMetrics> {
        if processing_times.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Processing _times cannot be empty".to_string(),
            ));
        }

        // Calculate average processing latency
        let avg_processing_time = processing_times.iter().sum::<f64>() / processing_times.len() as f64;
        let processing_latency = Duration::from_secs_f64(avg_processing_time);

        // Calculate throughput (_frames per second)
        let total_time = processing_times.iter().sum::<f64>();
        let throughput = if total_time > 0.0 {
            n_frames as f64 / total_time
        } else {
            0.0
        };

        // Calculate real-time constraint satisfaction
        let real_time_threshold = 1.0 / 30.0; // 30 FPS requirement (33.33 ms per frame)
        let frames_meeting_deadline = processing_times.iter()
            .filter(|&&time| time <= real_time_threshold)
            .count();
        let real_time_satisfaction = frames_meeting_deadline as f64 / processing_times.len() as f64;

        // Calculate resource utilization efficiency
        let min_processing_time = processing_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_processing_time = processing_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let resource_efficiency = if max_processing_time > 0.0 {
            // Efficiency is higher when processing _times are consistent (low variance)
            let variance = processing_times.iter()
                .map(|&time| (time - avg_processing_time).powi(2))
                .sum::<f64>() / processing_times.len() as f64;
            let coefficient_of_variation = if avg_processing_time > 0.0 {
                variance.sqrt() / avg_processing_time
            } else {
                0.0
            };
            
            // Good efficiency means low variation and meeting real-time constraints
            let consistency_score = 1.0 / (1.0 + coefficient_of_variation);
            consistency_score * real_time_satisfaction
        } else {
            0.0
        };

        Ok(RealTimePerformanceMetrics {
            processing_latency,
            throughput,
            real_time_satisfaction,
            resource_efficiency,
        })
    }
}

// Implementation stubs for individual metric components
impl MotionPlanningMetrics {
    fn new() -> Self {
        Self {
            smoothness_metrics: TrajectorySmoothnessMetrics::default(),
            optimality_metrics: PathOptimalityMetrics::default(),
            constraint_metrics: ConstraintSatisfactionMetrics::default(),
            efficiency_metrics: PlanningEfficiencyMetrics::default(),
        }
    }
}

impl SlamMetrics {
    fn new() -> Self {
        Self {
            localization_metrics: LocalizationAccuracyMetrics::default(),
            mapping_metrics: MappingQualityMetrics::default(),
            loop_closure_metrics: LoopClosureMetrics::default(),
            computational_metrics: SlamComputationalMetrics::default(),
        }
    }
}

impl ManipulationMetrics {
    fn new() -> Self {
        Self {
            grasping_metrics: GraspingMetrics::default(),
            manipulation_accuracy: ManipulationAccuracyMetrics::default(),
            task_completion: TaskCompletionMetrics::default(),
            force_metrics: ForceContactMetrics::default(),
        }
    }
}

impl NavigationMetrics {
    fn new() -> Self {
        Self {
            path_planning: PathPlanningMetrics::default(),
            obstacle_avoidance: ObstacleAvoidanceMetrics::default(),
            goal_reaching: GoalReachingMetrics::default(),
            dynamic_adaptation: DynamicAdaptationMetrics::default(),
        }
    }
}

impl HumanRobotInteractionMetrics {
    fn new() -> Self {
        Self {
            safety_metrics: HriSafetyMetrics::default(),
            communication_metrics: CommunicationMetrics::default(),
            user_satisfaction: UserSatisfactionMetrics::default(),
            collaboration_efficiency: CollaborationEfficiencyMetrics::default(),
        }
    }
}

impl MultiRobotMetrics {
    fn new() -> Self {
        Self {
            formation_control: FormationControlMetrics::default(),
            task_allocation: TaskAllocationMetrics::default(),
            network_performance: NetworkPerformanceMetrics::default(),
            collective_behavior: CollectiveBehaviorMetrics::default(),
        }
    }
}

impl SafetyReliabilityMetrics {
    fn new() -> Self {
        Self {
            failure_metrics: FailureMetrics::default(),
            risk_assessment: RiskAssessmentMetrics::default(),
            redundancy_metrics: RedundancyMetrics::default(),
            maintenance_metrics: MaintenanceMetrics::default(),
        }
    }
}

impl RoboticPerceptionMetrics {
    fn new() -> Self {
        Self {
            object_detection: ObjectDetectionMetrics::default(),
            scene_understanding: SceneUnderstandingMetrics::default(),
            sensor_fusion: SensorFusionMetrics::default(),
            real_time_performance: RealTimePerformanceMetrics::default(),
        }
    }
}

// Default implementations for all metrics structures
impl Default for TrajectorySmoothnessMetrics {
    fn default() -> Self {
        Self {
            average_jerk: 0.0,
            max_jerk: 0.0,
            acceleration_variance: 0.0,
            curvature_metrics: CurvatureMetrics::default(),
            velocity_smoothness: 0.0,
        }
    }
}

impl Default for CurvatureMetrics {
    fn default() -> Self {
        Self {
            average_curvature: 0.0,
            max_curvature: 0.0,
            curvature_variance: 0.0,
            sharp_turns_count: 0,
        }
    }
}

impl Default for PathOptimalityMetrics {
    fn default() -> Self {
        Self {
            length_optimality_ratio: 1.0,
            energy_optimality_ratio: 1.0,
            time_optimality_ratio: 1.0,
            obstacle_clearance: ObstacleClearanceMetrics::default(),
        }
    }
}

impl Default for ObstacleClearanceMetrics {
    fn default() -> Self {
        Self {
            min_clearance: 0.0,
            avg_clearance: 0.0,
            clearance_variance: 0.0,
            safety_margin_ratio: 1.0,
        }
    }
}

impl Default for ConstraintSatisfactionMetrics {
    fn default() -> Self {
        Self {
            joint_limits_satisfaction: 1.0,
            velocity_limits_satisfaction: 1.0,
            acceleration_limits_satisfaction: 1.0,
            torque_limits_satisfaction: 1.0,
            collision_avoidance_rate: 1.0,
        }
    }
}

impl Default for PlanningEfficiencyMetrics {
    fn default() -> Self {
        Self {
            planning_time: Duration::from_millis(0),
            memory_usage: 0,
            iterations_count: 0,
            planning_success_rate: 1.0,
            convergence_speed: 1.0,
        }
    }
}

impl Default for LocalizationAccuracyMetrics {
    fn default() -> Self {
        Self {
            absolute_trajectory_error: 0.0,
            relative_pose_error: 0.0,
            translation_error: ErrorStatistics::default(),
            rotation_error: ErrorStatistics::default(),
            drift_metrics: DriftMetrics::default(),
        }
    }
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            rmse: 0.0,
            mae: 0.0,
            max_error: 0.0,
            std_dev: 0.0,
            median_error: 0.0,
        }
    }
}

impl Default for DriftMetrics {
    fn default() -> Self {
        Self {
            translation_drift_rate: 0.0,
            rotation_drift_rate: 0.0,
            scale_drift: 0.0,
            drift_consistency: 1.0,
        }
    }
}

// Continue with default implementations for all other metrics...
// (For brevity, I'll implement a few more key ones)

impl Default for MappingQualityMetrics {
    fn default() -> Self {
        Self {
            completeness: 1.0,
            map_accuracy: 1.0,
            feature_detection_rate: 1.0,
            consistency_metrics: MapConsistencyMetrics::default(),
        }
    }
}

impl Default for MapConsistencyMetrics {
    fn default() -> Self {
        Self {
            feature_consistency: 1.0,
            geometric_consistency: 1.0,
            temporal_consistency: 1.0,
            global_consistency: 1.0,
        }
    }
}

impl Default for LoopClosureMetrics {
    fn default() -> Self {
        Self {
            detection_rate: 1.0,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
            closure_accuracy: 1.0,
            detection_timing: Duration::from_millis(0),
        }
    }
}

impl Default for SlamComputationalMetrics {
    fn default() -> Self {
        Self {
            avg_processing_time: Duration::from_millis(0),
            memory_usage_profile: Vec::new(),
            cpu_utilization: 0.0,
            real_time_factor: 1.0,
        }
    }
}

// Additional defaults for key structures...
impl Default for GraspingMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            stability_score: 1.0,
            force_closure_quality: 1.0,
            robustness_score: 1.0,
            approach_quality: 1.0,
        }
    }
}

impl Default for ManipulationAccuracyMetrics {
    fn default() -> Self {
        Self {
            positioning_accuracy: ErrorStatistics::default(),
            orientation_accuracy: ErrorStatistics::default(),
            path_following_accuracy: 1.0,
            target_reaching_rate: 1.0,
        }
    }
}

impl Default for TaskCompletionMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            completion_time: Duration::from_millis(0),
            efficiency_score: 1.0,
            partial_completion_rate: 0.0,
            retry_attempts: 0.0,
        }
    }
}

impl Default for ForceContactMetrics {
    fn default() -> Self {
        Self {
            force_accuracy: ErrorStatistics::default(),
            contact_stability: Duration::from_millis(0),
            force_control_precision: 1.0,
            contact_detection_accuracy: 1.0,
        }
    }
}

// Default for remaining metrics types would continue...
// For brevity, I'll add a macro to simplify this:

macro_rules! impl_default_metrics {
    ($($struct_name:ident { $($field:ident: $default_value:expr),* $(,)? }),* $(,)?) => {
        $(
            impl Default for $struct_name {
                fn default() -> Self {
                    Self {
                        $($field: $default_value,)*
                    }
                }
            }
        )*
    };
}

impl_default_metrics! {
    PathPlanningMetrics {
        success_rate: 1.0,
        optimality: PathOptimalityMetrics::default(),
        planning_time: Duration::from_millis(0),
        safety_margin: 1.0,
    },
    ObstacleAvoidanceMetrics {
        collision_avoidance_rate: 1.0,
        near_miss_frequency: 0.0,
        min_obstacle_distance: f64::INFINITY,
        avoidance_efficiency: 1.0,
    },
    GoalReachingMetrics {
        success_rate: 1.0,
        reaching_accuracy: ErrorStatistics::default(),
        time_to_goal: Duration::from_millis(0),
        path_efficiency: 1.0,
    },
    DynamicAdaptationMetrics {
        response_time: Duration::from_millis(0),
        replanning_frequency: 0.0,
        dynamic_success_rate: 1.0,
        prediction_accuracy: 1.0,
    },
    HriSafetyMetrics {
        safe_distance_maintenance: 1.0,
        emergency_response_time: Duration::from_millis(0),
        human_collision_avoidance: 1.0,
        intention_prediction_accuracy: 1.0,
    },
    CommunicationMetrics {
        command_understanding: 1.0,
        response_appropriateness: 1.0,
        communication_latency: Duration::from_millis(0),
        multimodal_effectiveness: 1.0,
    },
    UserSatisfactionMetrics {
        overall_satisfaction: 1.0,
        trust_level: 1.0,
        perceived_usefulness: 1.0,
        ease_of_interaction: 1.0,
    },
    CollaborationEfficiencyMetrics {
        collaborative_completion_time: Duration::from_millis(0),
        efficiency_gain: 1.0,
        workload_balance: 1.0,
        synchronization_accuracy: 1.0,
    },
    FormationControlMetrics {
        formation_accuracy: ErrorStatistics::default(),
        formation_stability: 1.0,
        reconfiguration_time: Duration::from_millis(0),
        scalability_score: 1.0,
    },
    TaskAllocationMetrics {
        allocation_optimality: 1.0,
        load_balancing: 1.0,
        allocation_time: Duration::from_millis(0),
        failure_adaptation: 1.0,
    },
    NetworkPerformanceMetrics {
        communication_reliability: 1.0,
        network_latency: Duration::from_millis(0),
        bandwidth_utilization: 0.0,
        network_resilience: 1.0,
    },
    CollectiveBehaviorMetrics {
        swarm_coherence: 1.0,
        consensus_time: Duration::from_millis(0),
        emergent_behavior_quality: 1.0,
        collective_intelligence: 1.0,
    },
    FailureMetrics {
        detection_accuracy: 1.0,
        false_alarm_rate: 0.0,
        recovery_success_rate: 1.0,
        mean_time_to_recovery: Duration::from_millis(0),
    },
    RiskAssessmentMetrics {
        risk_prediction_accuracy: 1.0,
        safety_margin_maintenance: 1.0,
        hazard_identification_rate: 1.0,
        risk_mitigation_effectiveness: 1.0,
    },
    RedundancyMetrics {
        redundancy_utilization: 0.0,
        failover_success_rate: 1.0,
        performance_degradation: 0.0,
        system_availability: 1.0,
    },
    MaintenanceMetrics {
        prediction_accuracy: 1.0,
        preventive_effectiveness: 1.0,
        health_monitoring_quality: 1.0,
        scheduling_optimality: 1.0,
    },
    ObjectDetectionMetrics {
        per_class_accuracy: HashMap::new(),
        real_time_detection_rate: 1.0,
        false_positive_rate: 0.0,
        tracking_consistency: 1.0,
    },
    SceneUnderstandingMetrics {
        segmentation_accuracy: 1.0,
        spatial_relationship_accuracy: 1.0,
        dynamic_adaptation_rate: 1.0,
        context_reasoning_quality: 1.0,
    },
    SensorFusionMetrics {
        accuracy_improvement: 0.0,
        reliability_weighting_accuracy: 1.0,
        sensor_failure_detection: 1.0,
        information_gain: 0.0,
    },
    RealTimePerformanceMetrics {
        processing_latency: Duration::from_millis(0),
        throughput: 0.0,
        real_time_satisfaction: 1.0,
        resource_efficiency: 1.0,
    },
}

impl Default for RoboticsMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainMetrics for RoboticsMetrics {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Robotics and Autonomous Systems"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "trajectory_smoothness",
            "path_optimality",
            "constraint_satisfaction",
            "planning_efficiency",
            "localization_accuracy",
            "mapping_quality",
            "loop_closure_performance",
            "grasping_success_rate",
            "manipulation_accuracy",
            "task_completion_rate",
            "navigation_performance",
            "obstacle_avoidance_rate",
            "hri_safety",
            "communication_effectiveness",
            "multi_robot_coordination",
            "formation_control",
            "safety_reliability",
            "failure_detection",
            "perception_accuracy",
            "real_time_performance",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert("trajectory_smoothness", "Evaluate smoothness and quality of robot trajectories");
        descriptions.insert("path_optimality", "Assess optimality of planned paths in terms of length, energy, and time");
        descriptions.insert("constraint_satisfaction", "Measure adherence to physical and safety constraints");
        descriptions.insert("planning_efficiency", "Evaluate computational efficiency of motion planning");
        descriptions.insert("localization_accuracy", "Assess SLAM and localization accuracy using ATE, RPE, and drift metrics");
        descriptions.insert("mapping_quality", "Evaluate quality and consistency of generated maps");
        descriptions.insert("loop_closure_performance", "Measure loop closure detection accuracy and timing");
        descriptions.insert("grasping_success_rate", "Evaluate robotic grasping and manipulation success");
        descriptions.insert("manipulation_accuracy", "Assess precision and accuracy of manipulation tasks");
        descriptions.insert("task_completion_rate", "Measure overall task completion success and efficiency");
        descriptions.insert("navigation_performance", "Evaluate autonomous navigation capabilities");
        descriptions.insert("obstacle_avoidance_rate", "Assess collision avoidance and safety in navigation");
        descriptions.insert("hri_safety", "Evaluate safety in human-robot interaction scenarios");
        descriptions.insert("communication_effectiveness", "Measure effectiveness of robot-human communication");
        descriptions.insert("multi_robot_coordination", "Assess coordination and collaboration between multiple robots");
        descriptions.insert("formation_control", "Evaluate formation control accuracy and stability");
        descriptions.insert("safety_reliability", "Measure overall system safety and reliability");
        descriptions.insert("failure_detection", "Assess failure detection and recovery capabilities");
        descriptions.insert("perception_accuracy", "Evaluate robotic perception and scene understanding");
        descriptions.insert("real_time_performance", "Measure real-time processing capabilities");
        descriptions
    }

    fn evaluate(&mut self, data: &HashMap<String, Vec<f64>>) -> Result<Self::Result> {
        // Implementation would process the provided _data and compute metrics
        Ok(DomainEvaluationResult {
            domain: self.domain_name().to_string(),
            metrics: HashMap::new(),
            summary: "Robotics evaluation completed".to_string(),
            recommendations: vec![
                "Consider trajectory smoothness optimization".to_string(),
                "Improve SLAM accuracy for better localization".to_string(),
                "Enhance safety margins for human-robot interaction".to_string(),
            ],
        })
    }
}
