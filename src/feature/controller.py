class FeatureController:
    def __init__(self, feature_extractors):
        self.feature_extractors = feature_extractors

    def extract_features_from_objects(self, objects):
        objects_features = {} 
        
        for extractor_name, extractor in self.feature_extractors.items():
            features = extractor.extract_features(objects)
            objects_features[extractor_name] = features

        return objects_features
