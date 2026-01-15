//
// LocationManager.swift
//
// Manages location services for capturing GPS coordinates during video recording.
// Provides current location and heading information for metadata.
//

import CoreLocation
import Foundation

@MainActor
class LocationManager: NSObject, ObservableObject {
  static let shared = LocationManager()

  @Published var currentLocation: CLLocation?
  @Published var currentHeading: CLHeading?
  @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined

  private let locationManager = CLLocationManager()

  override init() {
    super.init()
    locationManager.delegate = self
    locationManager.desiredAccuracy = kCLLocationAccuracyBest
    locationManager.distanceFilter = 10 // Update every 10 meters
    authorizationStatus = locationManager.authorizationStatus
  }

  func requestPermission() {
    locationManager.requestWhenInUseAuthorization()
  }

  func startUpdatingLocation() {
    guard authorizationStatus == .authorizedWhenInUse || authorizationStatus == .authorizedAlways else {
      return
    }
    locationManager.startUpdatingLocation()
    locationManager.startUpdatingHeading()
  }

  func stopUpdatingLocation() {
    locationManager.stopUpdatingLocation()
    locationManager.stopUpdatingHeading()
  }
}

extension LocationManager: CLLocationManagerDelegate {
  nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
    Task { @MainActor in
      authorizationStatus = manager.authorizationStatus
      if authorizationStatus == .authorizedWhenInUse || authorizationStatus == .authorizedAlways {
        startUpdatingLocation()
      }
    }
  }

  nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
    Task { @MainActor in
      currentLocation = locations.last
    }
  }

  nonisolated func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
    Task { @MainActor in
      currentHeading = newHeading
    }
  }

  nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
    NSLog("[Blindsighted] Location error: \(error.localizedDescription)")
  }
}
