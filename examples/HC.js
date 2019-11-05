Highcharts.chart('container', {
    chart: {
        //type: 'line',
        zoomType: 'xy'
    },
    title: {
        text: 'Monthly Average Temperature'
    },
    subtitle: {
        text: 'Source: WorldClimate.com'
    },
    xAxis: {
        type: 'category',
        categories: ['AA', 'BB', 'CC'],
        crosshair: true
    },
    yAxis: {
        title: {
            text: 'Temperature (°C)'
        },
        crosshair: true
    },
    plotOptions: {
        
    },
    credits:{
       enabled: false
  },
    tooltip: {
        shared: true,
        useHTML: true,
        formatter: function() {
        	console.log(this);
          return '<img src="' + this.points[0].point.wmap + '"></img>';
        },
        headerFormat: '<small>{point.key}</small><table>',
        pointFormat: '<tr><td style="color: {series.color}">{series.name}: </td>' +
            '<td style="text-align: right"><b>{point.y} EUR</b></td></tr>',
        footerFormat: '</table>',
        valueDecimals: 2
    },
    series: [{
        name: 'Tokyo',
        data: [
        	{y:7.0, wmap:'data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7'},
          {y:6.9, wmap:'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='},
          {y:9.5, wmap:'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='}
        ]
    }],
    boost: {
        useGPUTranslations: true,
        usePreallocated: true
    },
    annotations: [{
        labelOptions: {
            backgroundColor: 'rgba(255,255,255,0.5)',
            verticalAlign: 'top',
            y: 15
        },
        labels: [{
            point: {
                xAxis: 100,
                yAxis: 100,
                x: 100,
                y: 100
            },
            useHTML: true,
            text: 'Arbois<img src="AAA"></img>'
        }]
    }]
});
