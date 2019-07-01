import React from 'react';
import HeatMap from './components/heatmap/HeatMap'
import Collapsible from 'react-collapsible'

class ModelOutput extends React.Component {
  render() {

    const { outputs } = this.props;

    // TODO: `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    var labels = outputs['labels'];
    var alphas = outputs['alphas'];
    var mentions = outputs['mentions'];
    // This is a 1D attention array, which we need to make into a 2D matrix to use with our heat
    // map component.
    // var attention_data = outputs['attention_data'].map(x => [x]);
    // This is a 2D attention matrix.
    // var matrix_attention_data = outputs['matrix_attention_data'];
    // Labels for our 2D attention matrix, and the rows in our 1D attention array.
    // var column_labels = outputs['column_labels'];
    // var row_labels = outputs['row_labels'];

    // This is how much horizontal space you'll get for the row labels.  Not great to have to
    // specify it like this, or with this name, but that's what we have right now.
    var xLabelWidth = "70px";
    let rows = [];
    for (var i = 0; i < mentions.length; i++) {
      let rowID = `row${i}`
      let cell = []
      let cell1ID = `cell${i}-0`
      let cell2ID = `cell${i}-1`
      cell.push(<td key={cell1ID} id={cell1ID}>{alphas[i]}</td>)
      cell.push(<td key={cell2ID} id={cell2ID}>{mentions[i]}</td>)
      rows.push(<tr key={i} id={rowID}>{cell}</tr>)
    }

    let predictions = [];
    for (var i = 0; i < labels.length; i++) {
      let rowID = `row${i}`
      let cell = []
      let cell1ID = `cell${i}-0`
      let cell2ID = `cell${i}-1`
      cell.push(<td key={cell1ID} id={cell1ID}>{labels[i][0]}</td>)
      cell.push(<td key={cell2ID} id={cell2ID}>{labels[i][1]}</td>)
      predictions.push(<tr key={i} id={rowID}>{cell}</tr>)
    }


    return (
      <div className="model__content">

        {/*
         * TODO: This is where you display your output.  You can show whatever you want, however
         * you want.  We've got a few examples, of text-based output, and of visualizing model
         * internals using heat maps.
         */}

        <div className="form__field">
          <label>Predictions</label>
          <div className="container">
            <div className="row">
              <div className="col s12 board">
                <table id="predictions">
                  <tbody>
                    {predictions}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <div className="form__field">
          <label>Attention scores</label>
          <div className="container">
            <div className="row">
              <div className="col s12 board">
                <table id="attention-scores">
                  <tbody>
                    {rows}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

      </div>
    );
  }
}

export default ModelOutput;
