"""
Deepsort:
    mean shitf: look the object in its neighborhood
    optical flow: Predict direction movement based on position and velocity, Lucas Kanaader
    kalman filter: Predict the future state of an object base on the current data (gaussian distribution, steady velocity)
    Simple Online Realtime Tracking
        Detection
        Estimation
        Association - IoU (HUNGARIAN ALGORITHM)
        Track identity creation and destruction
    Deep - Measurement to track Association (relate measure and existing track Mahanobis distance)
"""

import numpy as np

from src.utils.cmc.ecc import ECC
from src.utils.matching import NearestNeighborDistanceMetric
from src.utils.ops import xyxy2tlwh

from src.tracking.sort import linear_assignment
from src.tracking.track import Track
from src.utils.matching import chi2inv95
from src.tracking.sort import iou_matching
from src.tracking.tracking import Tracking

class StrongSORT(Tracking):

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric=None,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=1,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        self.model = None
        self.max_iou_dist=max_iou_dist
        self.max_age=max_age
        self.n_init=n_init
        self.mc_lambda=mc_lambda
        self.ema_alpha=ema_alpha

        self.cmc = ECC()
        self.metric = metric

        if metric is None:
            self.metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        
        self.tracks = []
        self._next_id = 1


    def predict_detection(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def update(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([np.arange(len(dets)).reshape(-1, 1),dets])
        xyxy = dets[:, 3:7]
        rest = dets[:, :3]

        if len(self.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        rng = np.random.default_rng(2021)
        features = rng.random((len(dets), 524))

        tlwh = xyxy2tlwh(xyxy)
        detections = np.concatenate([rest, tlwh, features],axis=1)

        # update tracker
        self.predict_detection()
        self._update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([id], [cls], [conf], [x1, y1, x2, y2])).reshape(1, -1)#[det_ind]
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.empty((0,8)) #np.array([[]])
    
    def _update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self.associate(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def _initiate_track(self, detection):
        self.tracks.append(
            Track(
                detection,
                self._next_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
            )
        )
        self._next_id += 1


    def associate(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i,7:] for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])

            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections