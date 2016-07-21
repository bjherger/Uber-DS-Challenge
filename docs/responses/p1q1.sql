SELECT
  PERCENTILE_CONT(.9)
    WITHIN GROUP (ORDER BY trips.actual_eta-predicted_eta)
		AS 90th_percentile
	FROM trips
		LEFT OUTER JOIN cities
			WHERE trips.city_id == cities.city_id
  WHERE cities.city_name IN ('Qarth', 'Meereen')
    AND trips.status == 'completed'
    AND trips.request_at > (CURRENT_TIMESTAMP- INTERVAL '10 days');
