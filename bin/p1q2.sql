SELECT signups_enhanced.day_of_week, AVG(rode_in_first_week::int)
  FROM

  -- Create sub-table with one row for every rider who signed up, with rode_in_first_week metric
   ( SELECT events.*
        EXTRACT( DOW FROM _ts) AS day_of_week
        -- Actually compute rode_in_first_week metric
        -- Check if user has a ride
        (MIN(trips.request_at) IS NOT NULL
         -- First ride within 168 hours
        AND MIN(trips.request_at) <= MIN(events._ts) + INTERVAL '168 hours'
        -- No rides before sign up
        AND MIN(trips.request_at) >= MIN(events._ts))
          AS rode_in_first_week
      FROM trips
      LEFT OUTER JOIN

        --   Create sub-table with every rider's first completed trip
        (SELECT DISTINCT ON (trips.client_id) trips.client_id, request_at
          FROM trips
            WHERE trips.status == 'completed'
          ORDER BY trips.request_at ASC
        ) AS first_completed_trips

        WHERE events.rider_id  == first_completed_trips.client_id
        AND event_name == 'sign_up_success'
    ) AS signups_enhanced

    GROUP BY signups_enhanced.day_of_week
    WHERE EXTRACT(WEEK FROM signup_ts) == 1
      AND EXTRACT(YEAR FROM signup_ts) == 2016;
      AND city_name IN ('Qarth', 'Meereen');
