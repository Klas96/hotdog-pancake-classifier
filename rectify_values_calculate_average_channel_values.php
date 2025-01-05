<?php

function rectify_value__calculate_avrage_channel_values($path = 0){
	$conf = [
		'width' => 64,
		'height' => 64
	];

	#$content = file_get_contents($path);
	for ($x=0; $x < $conf['width']; $x++){
		for ($y=0; $y < $conf['height']; $y++){
			for ($ch=0; $ch < 3; $ch++){
				echo 'w: ' . $x . ', Y: ' . $y . ', CH: ' . $ch . PHP_EOL;
			}
		}
	}
}
scale_image();
